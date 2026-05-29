import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')
authors_json_path = os.path.join(script_dir, 'authors_info_final.json')

# 所有 mapping 阶段产出的 csv/png 都放这个子文件夹
MAPPING_OUTPUT_DIR = os.path.join(script_dir, 'mapping_outputs')
os.makedirs(MAPPING_OUTPUT_DIR, exist_ok=True)

# --- Domain 映射 (与 HF 官方 task taxonomy 对齐, 见 huggingface.co/tasks) ---
# 注: graph-ml 在 HF 上单独归为 "Other", 论文里 6 个 domain 没有, 会通过 fillna 落到 'Other'
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

# 反转映射
task_to_domain = {task: domain for domain, tasks in domain_to_tasks.items() for task in tasks}


def process_sort_by(sort_by):
    print(f"\n{'='*60}")
    print(f"处理: {sort_by}")
    print(f"{'='*60}")

    input_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条")

    # --- 映射 domain ---
    df['domain'] = df['pipeline_content'].map(task_to_domain).fillna('Other')

    # 质量检查: 'Other' 占比 + 未映射 pipeline 详情
    other_count = (df['domain'] == 'Other').sum()
    other_pct = other_count / len(df) * 100
    print(f"\n[质量检查] 'Other' domain: {other_count} 条 ({other_pct:.1f}%)")
    if other_count > 0:
        # 进一步分: 有 pipeline_content 但没映射 vs 完全没 pipeline_content
        no_pipeline = df['pipeline_content'].isna().sum()
        has_pipeline_but_unmapped = other_count - no_pipeline
        print(f"  - 完全没 pipeline_content (NaN): {no_pipeline}")
        print(f"  - 有 pipeline_content 但未在映射表里: {has_pipeline_but_unmapped}")
    unmapped = df[df['domain'] == 'Other']['pipeline_content'].dropna().value_counts()
    if len(unmapped) > 0:
        print(f"\n  未映射的 pipeline_content (top 20):")
        for tag, cnt in unmapped.head(20).items():
            print(f"    - {tag}: {cnt}")

    # 移动 domain 列到 pipeline_content 后面
    cols = df.columns.tolist()
    if 'pipeline_content' in cols:
        idx = cols.index('pipeline_content')
        cols.insert(idx + 1, cols.pop(cols.index('domain')))
        df = df[cols]

    # --- 映射 affiliation ---
    if os.path.exists(authors_json_path):
        with open(authors_json_path, 'r', encoding='utf-8') as f:
            authors_info = json.load(f)

        # 质量检查: 抓取失败率
        total_authors = len(authors_info)
        failed_authors = sum(1 for info in authors_info.values()
                             if isinstance(info, dict) and 'error' in info)
        print(f"\n[质量检查] authors_info_final.json: {total_authors} 个 author")
        print(f"  - 抓取失败 (含 'error'): {failed_authors} ({failed_authors/total_authors*100:.1f}%)")

        def _lookup_aff(a):
            if pd.isna(a):
                return 'unknown'
            info = authors_info.get(str(a))
            if info is None:
                return 'unknown'  # author 不在抓取结果里
            if isinstance(info, dict) and 'error' in info:
                return 'unknown'  # 抓取失败
            return info.get('affiliation', 'unknown')

        df['affiliation'] = df['author'].apply(_lookup_aff)

        # 移动 affiliation 列到 author 后面
        cols = df.columns.tolist()
        if 'author' in cols:
            idx = cols.index('author')
            cols.insert(idx + 1, cols.pop(cols.index('affiliation')))
            df = df[cols]
        print(f"\nAffiliation 分布 (模型级别):")
        print(df['affiliation'].value_counts().to_string())
        unknown_pct = (df['affiliation'] == 'unknown').sum() / len(df) * 100
        print(f"\n[质量检查] 'unknown' affiliation 占模型总数: {unknown_pct:.1f}%")
    else:
        print(f"⚠️ authors_info_final.json 不存在,跳过 affiliation 映射")

    # --- 保存 ---
    output_path = os.path.join(MAPPING_OUTPUT_DIR, f'data_with_domain_affiliation_{sort_by}_{SCHEME}.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ 已保存: {output_path}")

    # --- Domain 分布统计 ---
    category_domain = df.groupby(['category', 'domain']).size().reset_index(name='count')
    stats_path = os.path.join(MAPPING_OUTPUT_DIR, f'domain_distribution_{sort_by}_{SCHEME}.csv')
    category_domain.to_csv(stats_path, index=False)
    print(f"✅ Domain 分布已保存: {stats_path}")

    print(f"\nDomain 分布:")
    print(df['domain'].value_counts().to_string())

    # --- 画柱状图 (横向 x 轴标签, 不再倾斜) ---
    for cat in ['popular', 'unpopular']:
        cat_data = category_domain[category_domain['category'] == cat]
        if cat_data.empty:
            continue
        # 按 count 降序排,图更好看
        cat_data = cat_data.sort_values('count', ascending=False)
        plt.figure(figsize=(14, 6))
        plt.bar(cat_data['domain'], cat_data['count'],
                color='steelblue' if cat == 'popular' else 'orange')
        plt.xticks(rotation=0, fontsize=10)   # 横体, 不倾斜
        plt.yticks(fontsize=10)
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'{cat.capitalize()} Models by Domain (by {sort_by})', fontsize=13)
        # 给每个柱顶加数字
        for i, v in enumerate(cat_data['count']):
            plt.text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plot_path = os.path.join(MAPPING_OUTPUT_DIR, f'{cat}_domain_distribution_{sort_by}_{SCHEME}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {cat} domain 分布图已保存: {plot_path}")

    return df


if __name__ == '__main__':
    for sort_by in ['downloads', 'likes']:
        process_sort_by(sort_by)

    print(f"\n{'='*60}")
    print("✅ Domain + Affiliation 映射完成!")
    print(f"{'='*60}")