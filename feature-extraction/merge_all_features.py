import pandas as pd
import os

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, 'metadata_features.csv')
details_path = os.path.join(script_dir, 'model_detail_features.csv')
readme_path = os.path.join(script_dir, 'readme_features.csv')
output_path = os.path.join(script_dir, 'full_model_features.csv')

# --- 1. 加载数据 ---
print("正在加载数据...")
metadata = pd.read_csv(metadata_path)
details = pd.read_csv(details_path)
print(f"metadata: {len(metadata)} 条, {len(metadata.columns)} 列")
print(f"model details: {len(details)} 条, {len(details.columns)} 列")

# --- 2. 合并 metadata 和 model details ---
# metadata 的 id 格式: "Qwen_Qwen3.5-27B"（/ 替换为 _）
# details 的 model_id 格式: "Qwen/Qwen3.5-27B"（原始格式）
# 需要把 details 的 model_id 也替换 / 为 _ 来匹配
details['id'] = details['model_id'].str.replace('/', '_', regex=False)

# 检查匹配情况
metadata_ids = set(metadata['id'].unique())
details_ids = set(details['id'].unique())
intersection = metadata_ids.intersection(details_ids)
print(f"\nmetadata 唯一 ID: {len(metadata_ids)}")
print(f"details 唯一 ID: {len(details_ids)}")
print(f"匹配数量: {len(intersection)}")
print(f"metadata 中无 details: {len(metadata_ids - details_ids)}")
print(f"details 中无 metadata: {len(details_ids - metadata_ids)}")

# 去掉 details 里的 model_id 列（已经用 id 匹配了）
# 同时去掉 details 里跟 metadata 重复的时间列，避免 _x _y 冲突
details_drop_cols = ['model_id']
# details 里的 time_created_at 和 metadata 里的 created_time 是同一个东西，保留 metadata 的
if 'time_created_at' in details.columns:
    details_drop_cols.append('time_created_at')

details_to_merge = details.drop(columns=details_drop_cols, errors='ignore')
merged = pd.merge(metadata, details_to_merge, on='id', how='left')
print(f"\n合并后（metadata + details）: {len(merged)} 条, {len(merged.columns)} 列")

# --- 3. 合并 readme features ---
if os.path.exists(readme_path):
    readme = pd.read_csv(readme_path)
    print(f"readme features: {len(readme)} 条, {len(readme.columns)} 列")

    # readme 的 file_name 格式: "Qwen_Qwen3.5-27B_README"
    # 需要去掉 _README 后缀来匹配
    readme['id'] = readme['file_name'].str.replace('_README$', '', regex=True)

    # 避免列名冲突：readme 里的 has_arxiv 改名为 has_arxiv_link_in_readme
    if 'has_arxiv' in readme.columns:
        readme.rename(columns={'has_arxiv': 'has_arxiv_link_in_readme'}, inplace=True)

    readme_to_merge = readme.drop(columns=['file_name'], errors='ignore')
    merged = pd.merge(merged, readme_to_merge, on='id', how='left')

    # 填充 readme 相关的数值列为 0（没有 readme 的模型）
    readme_numeric_cols = [col for col in readme_to_merge.columns if col != 'id']
    for col in readme_numeric_cols:
        if merged[col].dtype in ['float64', 'int64']:
            merged[col] = merged[col].fillna(0).astype(int)

    print(f"合并后（+ readme）: {len(merged)} 条, {len(merged.columns)} 列")
else:
    print(f"⚠️ readme features 文件不存在，跳过: {readme_path}")
    print("（等 pandoc 转换和 readme 特征提取完成后再跑一次即可）")

# --- 4. 打印最终列名对照 ---
print(f"\n{'='*60}")
print(f"最终数据集: {len(merged)} 条, {len(merged.columns)} 列")
print(f"{'='*60}")
print("\n来自 metadata_features.csv 的列:")
meta_cols = [c for c in metadata.columns if c in merged.columns]
print(f"  {meta_cols}")
print("\n来自 model_detail_features.csv 的列:")
detail_cols = [c for c in details_to_merge.columns if c in merged.columns and c not in metadata.columns]
print(f"  {detail_cols}")
if os.path.exists(readme_path):
    print("\n来自 readme_features.csv 的列:")
    readme_cols = [c for c in readme_to_merge.columns if c in merged.columns and c not in metadata.columns and c not in details_to_merge.columns]
    print(f"  {readme_cols}")

# --- 5. 保存 ---
merged.to_csv(output_path, index=False)
print(f"\n✅ 最终数据已保存: {output_path}")