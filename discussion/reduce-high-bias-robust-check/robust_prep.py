import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
input_path = os.path.join(script_dir, '../../feature-extraction/full_model_features.csv')
authors_json_path = os.path.join(script_dir, '../../research-qs-analysis/cross-grouping/authors_info_final.json')


# --- Domain 映射 (与 cross-grouping/mapping_domain_affiliation.py 保持一致) ---
domain_to_tasks = {
    'Multimodal': [
        'audio-text-to-text', 'image-text-to-text',
        'image-text-to-image', 'image-text-to-video',
        'visual-question-answering', 'document-question-answering',
        'video-text-to-text', 'visual-document-retrieval',
        'any-to-any',
    ],
    'Computer Vision': [
        'depth-estimation', 'image-classification', 'object-detection',
        'image-segmentation', 'text-to-image', 'image-to-text',
        'image-to-image', 'image-to-video', 'unconditional-image-generation',
        'video-classification', 'text-to-video', 'zero-shot-image-classification',
        'mask-generation', 'zero-shot-object-detection',
        'text-to-3d', 'image-to-3d',
        'image-feature-extraction', 'keypoint-detection', 'video-to-video',
    ],
    'NLP': [
        'text-classification', 'token-classification',
        'table-question-answering', 'question-answering',
        'zero-shot-classification', 'translation', 'summarization',
        'feature-extraction', 'text-generation', 'fill-mask',
        'sentence-similarity', 'text-ranking',
    ],
    'Audio': [
        'text-to-speech', 'text-to-audio',
        'automatic-speech-recognition', 'audio-to-audio',
        'audio-classification', 'voice-activity-detection',
    ],
    'Tabular': [
        'tabular-classification', 'tabular-regression', 'time-series-forecasting',
    ],
    'Reinforcement Learning': [
        'reinforcement-learning', 'robotics',
    ],
}
task_to_domain = {task: domain for domain, tasks in domain_to_tasks.items() for task in tasks}


def add_category_column(df_sorted, anchor_col):
    df_sorted = df_sorted.reset_index(drop=True)
    n = len(df_sorted)
    popular_cut = int(n * 0.10)
    gap_cut = int(n * 0.20)
    positions = np.arange(n)
    df_sorted['category'] = np.where(
        positions < popular_cut, 'popular',
        np.where(positions < gap_cut, 'gap', 'unpopular')
    )
    cols = [c for c in df_sorted.columns if c != 'category']
    anchor_idx = cols.index(anchor_col)
    cols = cols[:anchor_idx + 1] + ['category'] + cols[anchor_idx + 1:]
    return df_sorted[cols]


def add_domain_column(df):
    df = df.copy()
    df['domain'] = df['pipeline_content'].map(task_to_domain).fillna('Other')
    cols = df.columns.tolist()
    if 'pipeline_content' in cols:
        idx = cols.index('pipeline_content')
        cols.insert(idx + 1, cols.pop(cols.index('domain')))
        df = df[cols]
    return df


def add_affiliation_column(df, authors_info):
    df = df.copy()

    def _lookup(a):
        if pd.isna(a):
            return 'unknown'
        info = authors_info.get(str(a))
        if info is None:
            return 'unknown'
        if isinstance(info, dict) and 'error' in info:
            return 'unknown'
        return info.get('affiliation', 'unknown')

    df['affiliation'] = df['author'].apply(_lookup)
    cols = df.columns.tolist()
    if 'author' in cols:
        idx = cols.index('author')
        cols.insert(idx + 1, cols.pop(cols.index('affiliation')))
        df = df[cols]
    return df


def main():
    print("正在加载数据...")
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条")

    # --- 跟主分析一样的过滤 ---
    df['created_time'] = pd.to_datetime(df['created_time'])
    cutoff_date = datetime(2026, 2, 28) - timedelta(days=180)
    df = df[df['created_time'] < cutoff_date]
    df = df[df['model_size_bytes'] >= 314572800]
    df = df[df['likes'] >= 2]
    df = df[df['if_repository'] == 1]
    df = df[df['if_restricted'] == 0]
    print(f"标准过滤后: {len(df)} 条")

    # --- 去掉 downloads top 1% (reviewer 关切的 pipeline-inflated outlier) ---
    threshold = df['downloads'].quantile(0.99)
    outliers = df[df['downloads'] > threshold]
    df_clean = df[df['downloads'] <= threshold]
    print(f"\nDownloads top 1% 阈值: {threshold:.0f}")
    print(f"去掉 outlier: {len(outliers)} 条")
    print(f"剩余数据: {len(df_clean)} 条")

    outlier_path = os.path.join(script_dir, f'outlier_models_{SCHEME}.csv')
    outliers[['id', 'downloads', 'likes']].to_csv(outlier_path, index=False)
    print(f"Outlier 列表已保存: {outlier_path}")

    # --- 加 domain ---
    df_clean = add_domain_column(df_clean)
    other_pct = (df_clean['domain'] == 'Other').mean() * 100
    print(f"\nDomain 映射完成 ('Other' 占 {other_pct:.1f}%)")

    # --- 加 affiliation ---
    if os.path.exists(authors_json_path):
        with open(authors_json_path, 'r', encoding='utf-8') as f:
            authors_info = json.load(f)
        df_clean = add_affiliation_column(df_clean, authors_info)
        unknown_pct = (df_clean['affiliation'] == 'unknown').mean() * 100
        print(f"Affiliation 映射完成 ('unknown' 占 {unknown_pct:.1f}%)")
    else:
        print(f"⚠️ {authors_json_path} 不存在,跳过 affiliation 映射")

    # --- 重新分组 ---
    df_by_dl = df_clean.sort_values(by='downloads', ascending=False)
    df_by_dl = add_category_column(df_by_dl, anchor_col='downloads')
    dl_path = os.path.join(script_dir, f'robust_filtered_by_downloads_{SCHEME}.csv')
    df_by_dl.to_csv(dl_path, index=False)
    print(f"\nBy downloads 已保存: {dl_path}")
    print(f"   category 分布: {df_by_dl['category'].value_counts().to_dict()}")

    df_by_lk = df_clean.sort_values(by='likes', ascending=False)
    df_by_lk = add_category_column(df_by_lk, anchor_col='likes')
    lk_path = os.path.join(script_dir, f'robust_filtered_by_likes_{SCHEME}.csv')
    df_by_lk.to_csv(lk_path, index=False)
    print(f"By likes 已保存: {lk_path}")
    print(f"   category 分布: {df_by_lk['category'].value_counts().to_dict()}")

    print(f"\n原始数据 vs Robustness 数据:")
    print(f"  原始 (过滤后): {len(df)} 条")
    print(f"  去掉 outlier: {len(df_clean)} 条 (减少 {len(outliers)} 条, {len(outliers)/len(df)*100:.2f}%)")


if __name__ == '__main__':
    main()
