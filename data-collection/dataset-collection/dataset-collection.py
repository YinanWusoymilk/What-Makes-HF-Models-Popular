import os
import requests
import json
import time
import yaml

class HuggingFaceDatasetsAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def get_all_datasets(self, max_results=1000):
        datasets = []
        next_page_url = f"/api/datasets?limit={max_results}"
        total_count = 0

        while next_page_url:
            try:
                response = self.session.get(self.base_url + next_page_url, timeout=60)
                if response.status_code != 200:
                    print(f"请求失败: Status {response.status_code}, 5秒后重试...")
                    time.sleep(5)
                    continue

                response_dict = response.json()
                if not response_dict:
                    break

                datasets.extend(response_dict)
                total_count += len(response_dict)
                print(f"已获取 {total_count} 个 datasets")

                next_link = response.links.get("next", {})
                next_page_url = next_link.get("url", "").replace(self.base_url, "") if next_link else None

                time.sleep(1)

            except Exception as e:
                print(f"请求异常: {e}, 10秒后重试...")
                time.sleep(10)
                continue

        return datasets


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(script_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['huggingface_keys'][0]
    hf_api = HuggingFaceDatasetsAPI(api_key)

    print("开始采集所有 HuggingFace datasets...")
    all_datasets = hf_api.get_all_datasets()
    num_datasets = len(all_datasets)

    from datetime import date
    today = date.today().strftime('%Y-%m-%d')

    json_path = os.path.join(script_dir, f'{today}_all_huggingface_datasets_{num_datasets}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_datasets, f, ensure_ascii=False)
    print(f"✅ JSON 已保存: {json_path} ({num_datasets} 条)")

    ids_path = os.path.join(script_dir, f'{today}_dataset_ids.txt')
    with open(ids_path, 'w', encoding='utf-8') as f:
        for ds in all_datasets:
            ds_id = ds.get('id', '')
            if ds_id:
                f.write(ds_id + '\n')
    print(f"✅ ID 列表已保存: {ids_path}")