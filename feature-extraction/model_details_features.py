import json
import csv
import os
from datetime import datetime


# Function to process the JSON data and output a CSV file.

def find_language(d):
    if "language" in d:
        return d["language"]
    for key, value in d.items():
        if isinstance(value, dict):
            lang = find_language(value)
            if lang:
                return lang
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    lang = find_language(item)
                    if lang:
                        return lang
    return ""

def find_author(d):
    if "author" in d:
        return d["author"]
    for key, value in d.items():
        if isinstance(value, dict):
            author = find_author(value)
            if author:
                return author
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    author = find_author(item)
                    if author:
                        return author
    return ""

#evaluation results on the metadata card 
def has_non_empty_results(attributes):
    # 检查 'model_index' 是否存在且 'results' 非空
    model_index = attributes.get('model-index')
    if model_index and isinstance(model_index, list):
        # 确保 model_index 是一个列表并且至少有一个条目包含非空的 'results'
        for item in model_index:
            # 检查 'results' 键是否存在且其值为非空列表
            if 'results' in item and len(item['results']) != 0:
                return 1
    return 0

def format_timestamp(timestamp):
    if timestamp:
        try:
            # 处理 Z 结尾的 ISO 时间格式
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error formatting timestamp: {e}")
            return timestamp  # 返回原始字符串以防格式化失败
    else:
        return "N/A"  # 对于缺失的时间戳，返回 "N/A"


def check_repository(attributes):
    # 检查是否存在错误信息，且内容为"Repository not found"
    if "error" in attributes and attributes["error"] == "Repository not found":
        return 0
    return 1

def check_if_restricted(attributes):
    error_message = attributes.get("error", "")
    if "is restricted" in error_message:
        return 1
    return 0


def calculate_model_size(size_tree):
    """从 size tree 数据计算模型总大小（bytes）"""
    if not isinstance(size_tree, dict):
        return 0
    if "error" in size_tree:
        return 0
    total = 0
    for path, size in size_tree.items():
        if isinstance(size, (int, float)):
            total += size
    return total


def process_json_to_csv_with_additional_fields(json_filepath, size_tree_filepath, csv_filepath):
    print("正在加载 model details（可能需要几分钟）...")
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"✅ 已加载 {len(data)} 个模型 details")

    print("正在加载 model size tree（可能需要几分钟）...")
    with open(size_tree_filepath, 'r', encoding='utf-8') as file:
        size_data = json.load(file)
    print(f"✅ 已加载 {len(size_data)} 个模型 size tree")

    csv_data = []
    for model_id, attributes in data.items():
        # 跳过错误条目（只有 error 字段的）
        if not isinstance(attributes, dict):
            continue

        num_root_file = 0
        modules = set()
        has_model = 0
        num_model_files = 0
        model_extensions = ['.bin', '.safetensors', '.pt', '.pth', '.ckpt', '.h5', '.onnx', '.gguf']

        for sibling in attributes.get("siblings", []):
            rfilename = sibling.get("rfilename", "")
            if 'model' in rfilename:
                has_model = 1  # Set the flag if 'model' is found in the filename.
            if any(rfilename.lower().endswith(ext) for ext in model_extensions):
                num_model_files += 1
            # We split the filename and get the first part only if it is a path.
            path_parts = rfilename.split('/')[0] if '/' in rfilename else None
            if path_parts:
                modules.add(path_parts)
            else:
                num_root_file += 1

        num_modules = len(modules)
        language = find_language(attributes)  # Use the recursive function to find 'language'
        has_specified_language = 1 if language else 0

        author = find_author(attributes)

        spaces = attributes.get("spaces", [])
        num_spaces = len(spaces)

        has_widgetData = 1 if attributes.get("widgetData") else 0

        has_config = 1 if attributes.get("config") else 0

        has_model_index = has_non_empty_results(attributes)

        time_created_at = format_timestamp(attributes.get("createdAt"))

        time_last_modified = format_timestamp(attributes.get("lastModified"))

        # 检查仓库是否存在
        if_repository = check_repository(attributes)

        # 检查模型是否受限
        if_restricted = check_if_restricted(attributes)

        # 计算模型总大小（bytes）
        size_tree = size_data.get(model_id, {})
        model_size = calculate_model_size(size_tree)

        csv_data.append([model_id, author, num_root_file, num_modules, has_model, num_model_files, num_spaces, has_widgetData, has_config, has_model_index, has_specified_language, language, time_created_at, time_last_modified, if_repository, if_restricted, model_size])

    with open(csv_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["model_id", "author", "num_root_file", "num_modules", "has_model", "num_model_files", "num_spaces", "has_widgetData",
             "has_config", "has_model_index_result", "has_language", "language", "time_created_at",
             "time_last_modified", "if_repository", "if_restricted", "model_size_bytes"])
        writer.writerows(csv_data)

    print(f"✅ CSV 已保存: {csv_filepath} ({len(csv_data)} 条)")

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
json_filepath = os.path.join(script_dir, '../data-collection/model-details-collection/model_details_final.json')
size_tree_filepath = os.path.join(script_dir, '../data-collection/model-size-collection/model_size_tree_final.json')
csv_filepath = os.path.join(script_dir, 'model_detail_features.csv')

# Call the function with the actual paths.
process_json_to_csv_with_additional_fields(json_filepath, size_tree_filepath, csv_filepath)