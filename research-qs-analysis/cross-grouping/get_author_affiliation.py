import threading
import requests
import json
import time
import yaml
import os
import pandas as pd
from queue import Queue, Empty
from tqdm import tqdm
from bs4 import BeautifulSoup


class HuggingFaceAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_author_page(self, author_name):
        """只试一次, 不退避重试。失败的 author 下次跑脚本时会自动重试 (load_processed_ids 不把它们标记成已完成)。"""
        url = f"{self.base_url}/{author_name}"
        try:
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                return {"error": "Author not found"}
            elif response.status_code == 429:
                return {"error": "Status 429"}
            else:
                return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


# 论文里定义的 8 个合法 affiliation 类别 (其他抓到的值都视作 noise)
VALID_AFFILIATIONS = {
    'company', 'university', 'non-profit', 'organization or individual',
    'community', 'unknown', 'classroom', 'government',
}


def parse_author_info(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    author_info = {}

    # Extract author name
    author_info['name'] = soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else 'N/A'

    # Extract affiliation: HF 页面里 'span.capitalize' 被很多地方复用 (activity feed
    # "updated/liked/published" 等),所以我们扫所有 span.capitalize, 取第一个落在白名单里的值。
    # 没找到就 fallback 到 'organization or individual'。
    affiliation = 'organization or individual'
    for tag in soup.find_all('span', class_='capitalize'):
        text = tag.text.strip().lower()
        if text in VALID_AFFILIATIONS:
            affiliation = text
            break
    author_info['affiliation'] = affiliation

    return author_info


# --- 全局限流冷却 ---
# 连续 N 个 429 后, 所有线程 sleep 这么久 (HF 用 token bucket, quota 耗尽要等 reset)
CONSECUTIVE_429_THRESHOLD = 20
GLOBAL_COOLDOWN_SECS = 180   # 3 分钟


def worker(api_key, author_queue, processed_ids, in_flight, processed_lock, write_lock,
           pbar, jsonl_path, interval, thread_id,
           rate_limit_state, rate_limit_lock):
    """rate_limit_state: dict 共享状态 {'consecutive_429': int, 'cooldown_until': float}"""
    hf_api = HuggingFaceAPI(api_key)

    while True:
        try:
            author = author_queue.get_nowait()
        except Empty:
            break

        # 检查全局冷却 (任何线程触发后所有线程一起等)
        with rate_limit_lock:
            now = time.time()
            wait = rate_limit_state['cooldown_until'] - now
        if wait > 0:
            pbar.write(f"⏸ [线程 {thread_id}] 全局冷却中, sleep {wait:.0f}s...")
            time.sleep(wait)

        with processed_lock:
            if author in processed_ids or author in in_flight:
                continue
            in_flight.add(author)

        actual_request_made = False
        try:
            result = hf_api.get_author_page(author)
            actual_request_made = True

            if isinstance(result, dict) and "error" in result:
                author_info = result
                # 维护连续 429 计数
                if '429' in str(result.get('error', '')):
                    with rate_limit_lock:
                        rate_limit_state['consecutive_429'] += 1
                        if rate_limit_state['consecutive_429'] >= CONSECUTIVE_429_THRESHOLD:
                            rate_limit_state['cooldown_until'] = time.time() + GLOBAL_COOLDOWN_SECS
                            rate_limit_state['consecutive_429'] = 0
                            pbar.write(f"🚨 连续 {CONSECUTIVE_429_THRESHOLD} 个 429,触发全局冷却 {GLOBAL_COOLDOWN_SECS}s")
                else:
                    with rate_limit_lock:
                        rate_limit_state['consecutive_429'] = 0
            else:
                author_info = parse_author_info(result)
                with rate_limit_lock:
                    rate_limit_state['consecutive_429'] = 0

        except Exception as e:
            author_info = {"error": f"SystemCrash: {str(e)}"}
            pbar.write(f"⚠️ [线程 {thread_id}] {author}: {str(e)}")
        finally:
            with write_lock:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({author: author_info}, ensure_ascii=False) + "\n")

            with processed_lock:
                processed_ids.add(author)
                in_flight.discard(author)

            pbar.update(1)
            if actual_request_made:
                time.sleep(interval)


def load_processed_ids(jsonl_path):
    """断点续传: 成功的、以及真正永久失败的 (404/401/403),都算"已完成";
    限流 / 网络错误 / 旧版 'Max retries exceeded' 都让它下次重试。
    """
    processed = set()
    if os.path.exists(jsonl_path):
        print("正在同步已有进度...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    for author, content in d.items():
                        if isinstance(content, dict) and "error" in content:
                            err = str(content["error"])
                            hard_errors = ["Author not found", "Status 401", "Status 403"]
                            if any(he in err for he in hard_errors):
                                processed.add(author)
                            # 否则 (429 / Max retries / 网络异常) 让下次重试
                        else:
                            processed.add(author)
                except:
                    continue
    print(f"✅ 断点续传:已跳过 {len(processed)} 个已完成的 author (含真正 404/401/403); 限流/网络错误的会重试。")
    return processed


def merge_jsonl_to_final_json(jsonl_path, final_json_path):
    print("正在合并为最终 JSON 文件...")
    all_data = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                all_data.update(obj)
            except:
                continue
    with open(final_json_path, 'w', encoding='utf-8') as out:
        json.dump(all_data, out, ensure_ascii=False)
    print(f"🎊 合并完成！共 {len(all_data)} 个 author。")


def main():
    # --- 路径设置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    SCHEME = '10-10-80'
    FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')
    output_dir = script_dir

    jsonl_path = os.path.join(output_dir, 'authors_info.jsonl')
    final_json_path = os.path.join(output_dir, 'authors_info_final.json')

    # 从 filtered 数据读取唯一 author 列表（downloads 和 likes 合并去重）
    authors = set()
    for sort_by in ['downloads', 'likes']:
        csv_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            authors.update(df['author'].dropna().unique())
    author_list = sorted(authors)
    print(f"唯一 author 数: {len(author_list)}")

    # 加载 config (复用 data-collection/model-size-collection 下的 huggingface_keys)
    config_path = os.path.join(script_dir, '../../data-collection/model-size-collection/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 断点续传
    processed_ids = load_processed_ids(jsonl_path)
    in_flight = set()
    processed_lock = threading.Lock()
    write_lock = threading.Lock()
    author_queue = Queue()

    for author in author_list:
        if author not in processed_ids:
            author_queue.put(author)

    pbar = tqdm(total=author_queue.qsize(), desc="👤 Author 采集进度", unit="author")

    # 全局限流状态 (所有线程共享)
    rate_limit_state = {'consecutive_429': 0, 'cooldown_until': 0.0}
    rate_limit_lock = threading.Lock()

    # 多线程采集 — 用所有 key (13个) burst 阶段尽快用完 quota, 然后靠全局冷却统一等
    INTERVAL = 3.0
    keys = config['huggingface_keys']
    threads = []
    for i, key in enumerate(keys):
        t = threading.Thread(target=worker, args=(
            key, author_queue, processed_ids, in_flight,
            processed_lock, write_lock, pbar, jsonl_path, INTERVAL, i + 1,
            rate_limit_state, rate_limit_lock,
        ))
        t.start()
        threads.append(t)
        time.sleep(0.1)

    for t in threads:
        t.join()

    pbar.close()

    # 合并
    if os.path.exists(jsonl_path):
        merge_jsonl_to_final_json(jsonl_path, final_json_path)

    # 统计 affiliation 分布
    with open(final_json_path, 'r', encoding='utf-8') as f:
        all_authors = json.load(f)

    affiliation_counts = {}
    for author, info in all_authors.items():
        aff = info.get('affiliation', 'unknown')
        affiliation_counts[aff] = affiliation_counts.get(aff, 0) + 1

    # 按数量排序打印
    print("\nAffiliation 分布:")
    for aff, count in sorted(affiliation_counts.items(), key=lambda x: -x[1]):
        print(f"  {aff}: {count}")

    # 保存
    aff_df = pd.DataFrame(list(affiliation_counts.items()), columns=['Affiliation', 'Count'])
    aff_df = aff_df.sort_values('Count', ascending=False)
    aff_df.to_csv(os.path.join(output_dir, 'affiliation_counts.csv'), index=False)
    print(f"✅ Affiliation 统计已保存")


if __name__ == '__main__':
    main()