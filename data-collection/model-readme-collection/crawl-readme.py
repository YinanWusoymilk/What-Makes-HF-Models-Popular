import threading, requests, json, time, yaml, random, os
from queue import Queue, Empty
from tqdm import tqdm


class HuggingFaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def download_readme(self, model_id):
        url = f"{self.base_url}/{model_id}/raw/main/README.md"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    return {"status": "success", "content": response.text}
                elif response.status_code == 429:
                    wait_time = 15 * (attempt + 1) + random.uniform(0, 5)
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 404:
                    return {"status": "no_readme"}
                else:
                    return {"status": "error", "error": f"Status {response.status_code}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {"status": "error", "error": str(e)}
        return {"status": "error", "error": "Max retries exceeded"}


def worker(api_key, model_queue, processed_ids, in_flight, processed_lock, write_lock,
           pbar, jsonl_path, readme_dir, interval, thread_id):
    hf_api = HuggingFaceAPI(api_key)

    while True:
        try:
            model_id = model_queue.get_nowait()
        except Empty:
            break

        with processed_lock:
            if model_id in processed_ids or model_id in in_flight:
                continue
            in_flight.add(model_id)

        actual_request_made = False
        try:
            result = hf_api.download_readme(model_id)
            actual_request_made = True

            # 如果成功，保存为 .md 文件
            if result["status"] == "success":
                safe_name = model_id.replace("/", "_")
                filepath = os.path.join(readme_dir, f"{safe_name}_README.md")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result["content"])
                # jsonl 里只记录状态，不存内容（内容已存为文件）
                log_entry = {"status": "success", "file": filepath}
            else:
                log_entry = result

        except Exception as e:
            log_entry = {"status": "error", "error": f"SystemCrash: {str(e)}"}
            pbar.write(f"⚠️ [线程 {thread_id}] {model_id}: {str(e)}")
        finally:
            with write_lock:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({model_id: log_entry}, ensure_ascii=False) + "\n")

            with processed_lock:
                processed_ids.add(model_id)
                in_flight.discard(model_id)

            pbar.update(1)
            if actual_request_made:
                time.sleep(interval)


def load_processed_ids(jsonl_path):
    """只跳过成功和不可恢复的错误，允许重试临时性错误"""
    processed = set()
    if os.path.exists(jsonl_path):
        print("正在同步已有进度...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    for mid, content in d.items():
                        if not isinstance(content, dict):
                            continue
                        status = content.get("status", "")
                        if status == "success" or status == "no_readme":
                            processed.add(mid)
                        elif status == "error":
                            err = str(content.get("error", ""))
                            hard_errors = ["Status 404", "Status 401", "Status 403", "Timeout"]
                            if any(he in err for he in hard_errors):
                                processed.add(mid)
                            # 429、连接错误等留给下次重试
                except:
                    continue
    print(f"✅ 断点续传：已跳过 {len(processed)} 个已完成/不可恢复的任务。")
    return processed


def main():
    # --- 基础路径设置 ---
    metadata_dir = 'data-collection/metadata-collection'
    readme_dir_base = 'data-collection/model-readme-collection'
    readme_dir = f'{readme_dir_base}/readmes'
    input_file = f'{metadata_dir}/2026-02-28_all_huggingface_models_2669509.json'
    jsonl_path = f'{readme_dir_base}/readme_progress.jsonl'

    # 设定爬取范围
    start_index, end_index = 0, 2700000
    # ------------------

    os.makedirs(readme_dir, exist_ok=True)
    with open(f'{readme_dir_base}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_ids = load_processed_ids(jsonl_path)
    in_flight = set()
    processed_lock = threading.Lock()
    write_lock = threading.Lock()
    model_queue = Queue()

    with open(input_file, 'r') as f:
        model_list = json.load(f)[start_index:end_index]

    for m in model_list:
        mid = m.get('modelId')
        if mid and mid not in processed_ids:
            model_queue.put(mid)

    pbar = tqdm(total=model_queue.qsize(), desc="📄 README 采集进度", unit="model")

    threads = []
    for i, key in enumerate(config['huggingface_keys']):
        t = threading.Thread(target=worker, args=(
            key, model_queue, processed_ids, in_flight,
            processed_lock, write_lock, pbar, jsonl_path,
            readme_dir, 0.8, i + 1
        ))
        t.start()
        threads.append(t)
        time.sleep(0.1)

    for t in threads:
        t.join()

    pbar.close()


if __name__ == '__main__':
    main()