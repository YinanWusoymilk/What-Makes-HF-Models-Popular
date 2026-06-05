"""
Step 0: 数据准备 (SCHEME = 20-10-70)

从 ../../../data-preproc-dist-analyze/filtered_data/ 读取该 scheme 的两份 by_downloads / by_likes CSV
(这两份 CSV 已经包含 category 列, 由主 pipeline 早期切好), 加 domain + affiliation 列, 保存到本目录。
"""
import pandas as pd
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '20-10-70'
FILTERED_DIR = os.path.join(script_dir, '../../../data-preproc-dist-analyze/filtered_data')
authors_json_path = os.path.join(script_dir, '../../../research-qs-analysis/cross-grouping/authors_info_final.json')

# --- Domain 映射 (与 cross-grouping/mapping_domain_affiliation.py 完全一致) ---
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


def add_domain(df):
    df = df.copy()
    df['domain'] = df['pipeline_content'].map(task_to_domain).fillna('Other')
    cols = df.columns.tolist()
    if 'pipeline_content' in cols:
        idx = cols.index('pipeline_content')
        cols.insert(idx + 1, cols.pop(cols.index('domain')))
        df = df[cols]
    return df


def add_affiliation(df, authors_info):
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
    print(f"=== Prep, SCHEME = {SCHEME} ===")

    with open(authors_json_path, 'r', encoding='utf-8') as f:
        authors_info = json.load(f)
    print(f"已加载 authors_info_final.json ({len(authors_info)} 个 author)")

    for sort_by in ['downloads', 'likes']:
        in_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
        df = pd.read_csv(in_path)
        print(f"\nby {sort_by}: {len(df)} 条 (category 分布: {df['category'].value_counts().to_dict()})")

        df = add_domain(df)
        other_pct = (df['domain'] == 'Other').mean() * 100
        print(f"  ✓ domain 列已加 ('Other' 占 {other_pct:.1f}%)")

        df = add_affiliation(df, authors_info)
        unknown_pct = (df['affiliation'] == 'unknown').mean() * 100
        print(f"  ✓ affiliation 列已加 ('unknown' 占 {unknown_pct:.1f}%)")

        out_path = os.path.join(script_dir, f'filtered_by_{sort_by}_{SCHEME}.csv')
        df.to_csv(out_path, index=False)
        print(f"  保存: {out_path}")


if __name__ == '__main__':
    main()
