import json
import time
import csv
from datetime import datetime
import requests
import yaml


class HuggingFaceAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = 'https://huggingface.co'
        self.interval = 1  # 控制请求频率，避免过快

    def set_interval(self, interval):
        self.interval = interval

    def get_all_models(self, max_results=1000):
        models = []
        next_page_url = f"/api/models?limit={max_results}"
        total_count = 0

        while next_page_url:
            print(f"正在处理 URL: {next_page_url}")
            response = requests.get(self.base_url + next_page_url)
            response_dict = json.loads(response.content)

            if not response_dict:
                break

            models.extend(response_dict)
            total_count += len(response_dict)
            print(f"已获取 {total_count} 个模型")

            next_link = response.links.get("next", {})
            next_page_url = next_link.get("url", "").replace(self.base_url, "") if next_link else None

            time.sleep(self.interval)

        return models


def json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as jf:
        data = json.load(jf)

    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Model ID', 'Other Details'])

        for model in data:
            model_id = model.get('modelId', '')
            other_details = json.dumps(model, ensure_ascii=False)
            writer.writerow([model_id, other_details])


if __name__ == '__main__':
    with open('data-collection/metadata-collection/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config.get("huggingface_key")
    hf_api = HuggingFaceAPI(api_key)

    all_models = hf_api.get_all_models()
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_count = len(all_models)

    # 相对路径保存到 metadata-collection 文件夹
    output_dir = 'data-collection/metadata-collection'
    json_file_name = f'{output_dir}/{current_date}_all_huggingface_models_{file_count}.json'
    csv_file_name = f'{output_dir}/{current_date}_all_huggingface_models_{file_count}.csv'

    # 保存 JSON
    with open(json_file_name, 'w', encoding='utf-8') as f:
        json.dump(all_models, f, indent=4)
    print(f"模型列表已保存到 JSON 文件: {json_file_name}")

    # 保存 CSV
    json_to_csv(json_file_name, csv_file_name)
    print(f"模型列表已保存到 CSV 文件: {csv_file_name}")

