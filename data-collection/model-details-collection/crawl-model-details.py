import threading, requests, json, time, yaml, random, os
from queue import Queue, Empty
from tqdm import tqdm

class HuggingFaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_model_info_by_id(self, model_id):
        url = f"{self.base_url}/api/models/{model_id}"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 15 * (attempt + 1) + random.uniform(0, 5)
                    time.sleep(wait_time)
                    continue
                return {"error": f"Status {response.status_code}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return {"error": str(e)}
        return {"error": "Max retries exceeded"}

def worker(api_key, model_queue, processed_ids, in_flight, processed_lock, write_lock, progress_bar, jsonl_path, interval, thread_id):
    hf_api = HuggingFaceAPI(api_key)

    while True:
        try:
            model_id = model_queue.get_nowait()
        except Empty:
            break

        actual_request_made = False 
        try:
            with processed_lock:
                if model_id in processed_ids or model_id in in_flight:
                    continue 
                in_flight.add(model_id)
            
            model_info = hf_api.get_model_info_by_id(model_id)
            actual_request_made = True

            # 【更新】：打印每一个错误（包括 404）
            if isinstance(model_info, dict) and "error" in model_info:
                err_msg = model_info["error"]
                progress_bar.write(f"❌ [线程 {thread_id}] 模型 {model_id} 失败: {err_msg}")

            with write_lock:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({model_id: model_info}, ensure_ascii=False) + "\n")
                with processed_lock:
                    processed_ids.add(model_id)

        except Exception as e:
            progress_bar.write(f"⚠️ [线程 {thread_id}] 系统错误 {model_id}: {str(e)}")
        finally:
            with processed_lock:
                in_flight.discard(model_id)
            progress_bar.update(1)
            
            if actual_request_made:
                time.sleep(interval)

def load_processed_ids(jsonl_path):
    processed = set()
    if os.path.exists(jsonl_path):
        print(f"正在扫描进度文件...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    for mid, details in d.items():
                        if isinstance(details, dict):
                            if "error" not in details:
                                processed.add(mid)
                            else:
                                err_msg = str(details["error"])
                                if any(s in err_msg for s in ["Status 404", "Status 401", "Status 403"]):
                                    processed.add(mid)
                except: continue
    print(f"✅ 断点续传：已跳过 {len(processed)} 个已完成任务。")
    return processed

def merge_jsonl_to_big_json(jsonl_path, final_json_path):
    print(f"正在将流水账合并为最终大 JSON...")
    tmp_path = final_json_path + ".tmp"
    seen = set()
    first = True
    with open(tmp_path, 'w', encoding='utf-8') as out:
        out.write("{\n")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except: continue
                
                for mid, details in obj.items():
                    if mid in seen: continue
                    if not first:
                        out.write(",\n")
                    out.write(f"  {json.dumps(mid, ensure_ascii=False)}: {json.dumps(details, ensure_ascii=False)}")
                    seen.add(mid)
                    first = False
        out.write("\n}")
    os.replace(tmp_path, final_json_path)
    print(f"🎊 合并完成！最终文件：{final_json_path} (当前累计 {len(seen)} 条数据)")



def main():
    metadata_dir = 'data-collection/metadata-collection'
    details_dir = 'data-collection/model-details-collection'
    input_file = f'{metadata_dir}/2026-02-28_all_huggingface_models_2669509.json'
    jsonl_path = f'{details_dir}/model_details.jsonl'
    final_json_path = f'{details_dir}/model_details_final.json'
    
    # 手动设置范围
    start_index, end_index = 1600000, 2700000 
    # ------------

    os.makedirs(details_dir, exist_ok=True)
    with open(f'{details_dir}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processed_ids = load_processed_ids(jsonl_path)
    in_flight = set() 
    processed_lock = threading.Lock()
    write_lock = threading.Lock()
    model_queue = Queue()

    with open(input_file, 'r') as f:
        model_list = json.load(f)[start_index:end_index]
    
    for model in model_list:
        mid = model.get('modelId')
        if mid and mid not in processed_ids:
            model_queue.put(mid)
    
    pbar = tqdm(total=model_queue.qsize(), desc="🚀 采集进度", unit="model")

    threads = []
    # 记录线程编号 i+1
    for i, key in enumerate(config['huggingface_keys']):
        t = threading.Thread(
            target=worker, 
            args=(key, model_queue, processed_ids, in_flight, processed_lock, write_lock, pbar, jsonl_path, 0.8, i+1)
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)

    for t in threads:
        t.join()
    
    pbar.close()

    # 【更新】：每次跑完这一批，立即生成/更新最终的大 JSON 文件
    merge_jsonl_to_big_json(jsonl_path, final_json_path)

if __name__ == '__main__':
    main()