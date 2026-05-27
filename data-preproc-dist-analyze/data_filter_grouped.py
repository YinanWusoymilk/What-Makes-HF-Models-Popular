import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(str(msg))


def add_category_column(df_sorted, anchor_col, popular_pct, gap_pct):
    """对已经排序好的 df 按行位置打标签:
        前 popular_pct%               -> 'popular'
        紧接着 gap_pct%                -> 'gap'
        其余                            -> 'unpopular'
    然后把 'category' 插到 anchor_col 列的右边。
    popular_pct / gap_pct 是百分数,例如 5、10、15。
    """
    df_sorted = df_sorted.reset_index(drop=True)
    n = len(df_sorted)
    popular_cut = int(n * popular_pct / 100)
    gap_cut = int(n * (popular_pct + gap_pct) / 100)
    positions = np.arange(n)
    df_sorted['category'] = np.where(
        positions < popular_cut, 'popular',
        np.where(positions < gap_cut, 'gap', 'unpopular')
    )
    cols = [c for c in df_sorted.columns if c != 'category']
    anchor_idx = cols.index(anchor_col)
    cols = cols[:anchor_idx + 1] + ['category'] + cols[anchor_idx + 1:]
    return df_sorted[cols]

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, '../feature-extraction/full_model_features.csv')
output_dir = os.path.join(script_dir, 'filtered_data')
os.makedirs(output_dir, exist_ok=True)

log(f"# 数据过滤汇总")
log(f"")
log(f"## 过滤步骤")
log(f"")
log(f"| 步骤 | 剩余条数 |")
log(f"| --- | --- |")

# 加载CSV文件
print("正在加载数据...")
df = pd.read_csv(input_path)
log(f"| 原始数据 ({len(df.columns)} 列) | {len(df)} |")

# 确保created_time列是datetime类型的
df['created_time'] = pd.to_datetime(df['created_time'])

# 计算180天前的日期（数据采集日期: 2026-02-28）
cutoff_date = datetime(2026, 2, 28) - timedelta(days=180)
cutoff_str = cutoff_date.strftime('%Y-%m-%d')

# 设置过滤条件，保留180天之前的记录
df = df[df['created_time'] < cutoff_date]
log(f"| created_time < {cutoff_str} (180天前) | {len(df)} |")

# 过滤掉 model_size 小于 300 MiB 的数据 (300 MiB = 314572800 bytes)
df = df[df['model_size_bytes'] >= 314572800]
log(f"| model_size >= 300 MiB | {len(df)} |")

# 过滤掉点赞量低于2的
df = df[df['likes'] >= 2]
log(f"| likes >= 2 | {len(df)} |")

# 删除if_repository列中等于0的行
df = df[df['if_repository'] == 1]
log(f"| if_repository == 1 | {len(df)} |")

# 过滤掉受限（gated）模型
df = df[df['if_restricted'] == 0]
log(f"| if_restricted == 0 | {len(df)} |")

# 多种分组方案: (popular%, gap%, unpopular%)
grouping_schemes = [
    (5, 10, 85),
    (10, 10, 80),
    (15, 10, 75),
    (20, 10, 70),
]

# 先排好序，避免每个分组方案都重复排序
df_sorted_by_likes = df.sort_values(by='likes', ascending=False)
df_sorted_by_downloads = df.sort_values(by='downloads', ascending=False)

log(f"")
log(f"最终数据: **{len(df)}** 条, {len(df.columns)} 列")
log(f"")
log(f"## 分组方案与 category 分布")
log(f"")
log(f"每格格式: `条数 (值范围)`,值范围对应该组的实际 likes/downloads 区间。")
log(f"")
log(f"| 排序依据 | 方案 (popular-gap-unpopular) | popular | gap | unpopular | 输出文件 |")
log(f"| --- | --- | --- | --- | --- | --- |")

def fmt_cell(df_cat, value_col, cat):
    sub = df_cat.loc[df_cat['category'] == cat, value_col]
    if sub.empty:
        return '0'
    return f"{len(sub)} ({int(sub.min())}–{int(sub.max())})"

for popular_pct, gap_pct, unpopular_pct in grouping_schemes:
    suffix = f"{popular_pct}-{gap_pct}-{unpopular_pct}"

    # by downloads
    df_downloads_cat = add_category_column(df_sorted_by_downloads, 'downloads', popular_pct, gap_pct)
    downloads_output = os.path.join(output_dir, f'filtered_model_data_by_downloads_{suffix}.csv')
    df_downloads_cat.to_csv(downloads_output, index=False)
    log(f"| downloads | {suffix} | "
        f"{fmt_cell(df_downloads_cat, 'downloads', 'popular')} | "
        f"{fmt_cell(df_downloads_cat, 'downloads', 'gap')} | "
        f"{fmt_cell(df_downloads_cat, 'downloads', 'unpopular')} | "
        f"`{os.path.basename(downloads_output)}` |")

    # by likes
    df_likes_cat = add_category_column(df_sorted_by_likes, 'likes', popular_pct, gap_pct)
    likes_output = os.path.join(output_dir, f'filtered_model_data_by_likes_{suffix}.csv')
    df_likes_cat.to_csv(likes_output, index=False)
    log(f"| likes | {suffix} | "
        f"{fmt_cell(df_likes_cat, 'likes', 'popular')} | "
        f"{fmt_cell(df_likes_cat, 'likes', 'gap')} | "
        f"{fmt_cell(df_likes_cat, 'likes', 'unpopular')} | "
        f"`{os.path.basename(likes_output)}` |")

# 写入 markdown
summary_path = os.path.join(output_dir, 'filter_summary.md')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(_log_lines) + '\n')
print(f"\n📝 汇总已写入: {summary_path}")
