import csv
import json
from datetime import datetime
import os

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(script_dir, '../data-collection/metadata-collection/2026-02-28_all_huggingface_models_2669509.json')
csv_file_path = os.path.join(script_dir, 'metadata_features.csv')
dataset_ids_dir = os.path.join(script_dir, '../data-collection/dataset-collection')

# 自动查找最新的 dataset_ids.txt
def find_dataset_ids_file(directory):
    for f in sorted(os.listdir(directory), reverse=True):
        if f.endswith('_dataset_ids.txt'):
            return os.path.join(directory, f)
    return None

# 读取所有Hugging Face数据集ID
def read_huggingface_dataset_ids(file_path):
    dataset_ids = set()
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset_ids.add(line)
        print(f"✅ 已加载 {len(dataset_ids)} 个 HuggingFace dataset ID")
    else:
        print("⚠️ 未找到 dataset ID 文件，match_huggingface_dataset 将全部标记为 0")
    return dataset_ids

# 检查dataset_content是否存在于Hugging Face数据集ID中
def check_dataset_in_huggingface(dataset_content, dataset_ids):
    if not dataset_ids or not dataset_content:
        return 0
    # 分割dataset_content以支持多个数据集ID的情况
    datasets = dataset_content.split(';')
    return any(dataset.strip() in dataset_ids for dataset in datasets)

# 读取Hugging Face数据集ID
dataset_ids_file = find_dataset_ids_file(dataset_ids_dir)
huggingface_dataset_ids = read_huggingface_dataset_ids(dataset_ids_file)


# 官方支持的库列表:https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts
official_libraries = [
    'adapter-transformers', 'allennlp', 'asteroid', 'bertopic', 'diffusers',
    'espnet', 'fairseq', 'fastai', 'fasttext', 'flair',
    'keras', 'keras-hub', 'ml-agents', 'mlx', 'nemo',
    'open_clip', 'paddlenlp', 'PaddleOCR', 'peft', 'pyannote-audio',
    'sample-factory', 'sentence-transformers', 'setfit', 'sklearn',
    'spacy', 'span-marker', 'speechbrain', 'stable-baselines3', 'stanza',
    'timm', 'transformers', 'transformers.js', 'unity-sentis', 'univa'
]

# 函数来检查库是否被官方支持
def check_if_supported(library, supported_list):
    return 1 if library in supported_list else 0


def extract_content(tags, prefix):
    contents = [tag.split(':', 1)[1] for tag in tags if tag.startswith(prefix) and ':' in tag]
    return ';'.join(contents)  # 使用分号连接所有内容

# Function to extract supported frameworks and libraries
def extract_additional_frameworks_and_libraries(tags, frameworks, primary_library):
    # 确保primary_library是一个字符串
    primary_library = primary_library or ''
    # Exclude the primary library from the search
    additional_frameworks = [framework for framework in frameworks if framework in tags and primary_library not in framework]
    return ';'.join(additional_frameworks), 1 if additional_frameworks else 0

# List of possible frameworks and libraries
# Fix: 修复了 'safetensors' 和 'stable-baselines3' 之间缺少逗号的 bug
frameworks_and_libraries = ['pytorch', 'tensorflow', 'jax', 'transformers',
                            'tensorboard', 'diffusers', 'peft', 'safetensors',
                            'stable-baselines3','onnx', 'ml-agents','gguf',
                            'sentence-transformers','keras','tf-keras','timm','flair',
                            'sample-factory', 'setfit','adapter-transformers', 'transformers.js',
                            'spacy','espnet','fastai','coreml','nemo',
                            'rust','joblib','fasttext','bertopic','mlx', 'mlx-llm',
                            'sklearn','speechbrain','OpenCLIP','open_clip','paddlepaddle',
                            'openvino','fairseq','optimum_graphcore', 'graphcore','tflite', 'asteroid',
                            'stanza','allennlp','paddlenlp','span-marker','optimum_habana', 'habana',
                            'pyannote','pyannote-audio','pythae','unity-sentis',
                            'litert','keras-hub','PaddleOCR','dduf','univa','llamafile','executorch']
#这个条目safetensors is difined as a library/frameworks in hugging face

# 读取JSON数据
print("正在加载元数据...")
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
print(f"原始数据: {len(data)} 个模型")

# 去重
seen_ids = set()
unique_data = []
duplicates = []
for entry in data:
    mid = entry.get('id', '')
    if mid not in seen_ids:
        seen_ids.add(mid)
        unique_data.append(entry)
    else:
        duplicates.append(mid)
data = unique_data
print(f"去重后: {len(data)} 个模型（发现 {len(duplicates)} 个重复 ID）")
if duplicates:
    print(f"重复的 ID: {duplicates}")

# 对数据按下载量从大到小排序
data.sort(key=lambda x: x.get('downloads', 0), reverse=True)

# 准备CSV文件
print("正在生成 CSV...")
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # 写入标题行
    writer.writerow(['id', 'model_id', 'downloads', 'likes', 'has_dataset', 'num_dataset', 'dataset_content', 'match_huggingface_dataset',
                     'has_arxiv', 'num_arxiv','arxiv_id', 'has_safetensors', 'has_quantized',
                     'has_license','license_content', 'has_region', 'region_content',
                     'has_space','has_pipeline_name', 'pipeline_content',
                     'has_primary_implementation_library_name','primary_implementation_library_content', 'if_supported_libraries',
                     'has_supported_additional_frameworks_Libraries', 'additional_supported_frameworks_libraries', 'created_time'
                     ])

    # 遍历JSON数据并提取信息
    for entry in data:
        # 提取id，并将"/"替换为"_"
        row_id = entry.get('id', '').replace('/', '_')
        row_downloads = entry.get('downloads', 0)
        row_likes = entry.get('likes', 0)
        model_number_id = entry.get('_id', '')

        # 检查tags以确定has_dataset和has_arxiv的值
        tags = entry.get('tags', [])


        # 根据tags中的内容设置标志位和提取具体内容
        row_has_dataset = 1 if any(tag.startswith('dataset:') for tag in tags) else 0
        dataset_content = extract_content(tags, 'dataset:') if row_has_dataset else ''
        num_dataset = len([dataset_id for dataset_id in dataset_content.split(';')]) if dataset_content else 0

        # 检查并设置has_huggingface_dataset
        match_huggingface_dataset = 1 if check_dataset_in_huggingface(dataset_content, huggingface_dataset_ids) else 0

        row_has_arxiv = 1 if any(tag.startswith('arxiv:') for tag in tags) else 0
        arxiv_content = extract_content(tags, 'arxiv:') if row_has_arxiv else ''
        num_arxiv = len([arxiv_id for arxiv_id in arxiv_content.split(';') if arxiv_id.strip()]) if arxiv_content else 0


        row_has_license = 1 if any(tag.startswith('license:') for tag in tags) else 0
        license_content = extract_content(tags, 'license:') if row_has_license else ''

        row_has_region = 1 if any(tag.startswith('region:') for tag in tags) else 0
        region_content = extract_content(tags, 'region:') if row_has_region else ''

        row_has_space = 1 if any(tag.startswith('has_space') for tag in tags) else 0

        row_has_safetensors = 1 if any(tag.startswith('safetensors') for tag in tags) else 0

        # 检查是否为量化模型（从 model id 和 tags 两个维度检查）
        quantization_keywords = ['gguf', 'gptq', 'awq', 'bnb', '4bit', '8bit', '4-bit', '8-bit', 'exl2', 'eetq', 'aqlm', 'hqq']
        # tags 是精确匹配，可以安全地把 'quantized' 加进去
        quantization_tag_keywords = quantization_keywords + ['quantized']
        # model id 是子串匹配，'quantized' 会误匹配 'unquantized'/'non-quantized'（含义相反），需要排除
        row_id_lower = row_id.lower()
        id_hit_specific = any(kw in row_id_lower for kw in quantization_keywords)
        id_hit_quantized = ('quantized' in row_id_lower
                            and 'unquantized' not in row_id_lower
                            and 'non-quantized' not in row_id_lower)
        has_quantized = 1 if (id_hit_specific or id_hit_quantized or
                              any(tag.lower() in quantization_tag_keywords for tag in tags)) else 0


        row_has_pipeline_tag = 1 if 'pipeline_tag' in entry else 0
        row_has_library_name = 1 if 'library_name' in entry else 0

        # 提取pipeline_tag和library_name的具体内容，如果不存在则留空
        pipeline_content = entry.get('pipeline_tag', '')
        library_content = entry.get('library_name', '')
        row_supported_library = check_if_supported(library_content, official_libraries)

        # Extract supported additional frameworks and libraries and check if any are supported
        supported_additional_frameworks_libraries, has_supported_additional_frameworks_libraries = extract_additional_frameworks_and_libraries(tags, frameworks_and_libraries, library_content)

        created_at = entry.get('createdAt', '')
        try:
            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
        except:
            created_time = created_at

        # 写入CSV文件
        writer.writerow([row_id, model_number_id, row_downloads, row_likes, row_has_dataset,num_dataset,dataset_content, match_huggingface_dataset,
                         row_has_arxiv, num_arxiv, arxiv_content, row_has_safetensors, has_quantized,
                         row_has_license, license_content, row_has_region, region_content,
                         row_has_space, row_has_pipeline_tag, pipeline_content, row_has_library_name, library_content, row_supported_library,
                         has_supported_additional_frameworks_libraries, supported_additional_frameworks_libraries, created_time])

print(f"✅ CSV 已保存: {csv_file_path} ({len(data)} 条)")