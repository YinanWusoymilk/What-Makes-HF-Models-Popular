import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_log_lines = []

def log(msg=''):
    print(msg)
    _log_lines.append(str(msg))

# --- 多种分组方案: (popular%, gap%, unpopular%) ---
SCHEMES = [
    (5, 10, 85),
    (10, 10, 80),
    (15, 10, 75),
    (20, 10, 70),
]

# --- 分组背景色 ---
COLOR_POPULAR = 'steelblue'
COLOR_GAP = 'gold'
COLOR_UNPOPULAR = 'lightgray'
BG_ALPHA = 0.25

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_plots_dir = os.path.join(script_dir, 'filtered_plots')
os.makedirs(output_plots_dir, exist_ok=True)

# --- 加载过滤后的数据 ---
# 4 个 scheme 过滤后的总集合相同（只是 category 列不同），任选其一读就行
print("正在加载过滤后的数据...")
df = pd.read_csv(os.path.join(script_dir, 'filtered_data', 'filtered_model_data_by_likes_10-10-80.csv'))

all_downloads = df['downloads'].tolist()
all_likes = df['likes'].tolist()

total = len(df)
dl_nonzero = [d for d in all_downloads if d > 0]
lk_nonzero = [l for l in all_likes if l > 0]

log(f"# 过滤后数据分布汇总")
log(f"")
log(f"## 总体统计")
log(f"")
log(f"- 总模型数: **{total}**")
log(f"- Downloads > 0: {len(dl_nonzero)} ({len(dl_nonzero)/total*100:.1f}%)")
log(f"- Likes > 0:     {len(lk_nonzero)} ({len(lk_nonzero)/total*100:.1f}%)")
log(f"")
log(f"| 指标 | median (非零) | mean (非零) | max |")
log(f"| --- | --- | --- | --- |")
log(f"| Downloads | {np.median(dl_nonzero):.0f} | {np.mean(dl_nonzero):.0f} | {max(all_downloads)} |")
log(f"| Likes     | {np.median(lk_nonzero):.0f} | {np.mean(lk_nonzero):.0f} | {max(all_likes)} |")
log(f"")
log(f"## 各分组方案的边界值")
log(f"")
# 注意:边界计算需要传入完整列表(含 0),才能跟 CSV 的位置切分对齐


def plot_distribution(values_all, xlabel, title, tick_vals, tick_labels, output_path,
                      popular_pct, gap_pct, unpopular_pct, metric_name=''):
    """画 log-scale KDE 分布图,带分组背景着色 + median 竖线 + 边界标注。
    边界用"按位置切"算,跟 data_filter_grouped.py 完全一致,保证图和 CSV 对得上。
    """
    arr = np.array(values_all, dtype=float)
    nonzero = arr[arr > 0]
    arr_log = np.log10(nonzero)

    # 按位置切分(跟 add_category_column 的逻辑一致)
    arr_sorted_desc = np.sort(arr)[::-1]
    n = len(arr_sorted_desc)
    popular_cut = int(n * popular_pct / 100)
    gap_cut = int(n * (popular_pct + gap_pct) / 100)
    popular_min = arr_sorted_desc[popular_cut - 1]   # popular 组的最小值
    gap_max = arr_sorted_desc[popular_cut]           # gap 组的最大值(可能 = popular_min 因 ties)
    gap_min = arr_sorted_desc[gap_cut - 1]           # gap 组的最小值
    unpopular_max = arr_sorted_desc[gap_cut]         # unpopular 组的最大值(可能 = gap_min 因 ties)

    p_popular_log = np.log10(popular_min)
    p_gap_log = np.log10(gap_min)
    p_median_val = np.percentile(nonzero, 50)
    p_median_log = np.log10(p_median_val)

    p_gap_percentile_val = 100 - popular_pct - gap_pct
    p_popular_percentile_val = 100 - popular_pct
    log(f"| {metric_name} | {popular_pct}-{gap_pct}-{unpopular_pct} | "
        f"P{p_popular_percentile_val}={int(popular_min)} | "
        f"P{p_gap_percentile_val}={int(gap_min)} | "
        f"{int(gap_min)}–{int(gap_max)} | "
        f"≤{int(unpopular_max)} |")

    kde = gaussian_kde(arr_log, bw_method=0.2)
    x = np.linspace(arr_log.min(), arr_log.max(), 1000)
    y = kde(x)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(arr_log.min(), p_gap_log, alpha=BG_ALPHA, color=COLOR_UNPOPULAR,
               label=f'unpopular ({unpopular_pct}%, ≤ {int(unpopular_max)})')
    ax.axvspan(p_gap_log, p_popular_log, alpha=BG_ALPHA, color=COLOR_GAP,
               label=f'gap ({gap_pct}%, {int(gap_min)}–{int(gap_max)})')
    ax.axvspan(p_popular_log, arr_log.max(), alpha=BG_ALPHA, color=COLOR_POPULAR,
               label=f'popular ({popular_pct}%, ≥ {int(popular_min)})')

    ax.plot(x, y, 'k-', linewidth=1.5)

    ax.axvline(p_median_log, color='black', linestyle='--', linewidth=1.2,
               label=f'Median: {p_median_val:.0f}')

    y_max = y.max()
    label_y_high = y_max * 1.15
    label_y_low = y_max * 0.95
    p_gap_percentile = 100 - popular_pct - gap_pct
    p_popular_percentile = 100 - popular_pct
    bbox_style = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='dimgray', alpha=0.9)
    arrow_style = dict(arrowstyle='-', color='dimgray', lw=0.5)
    ax.annotate(f'P{p_gap_percentile}={int(gap_min)}',
                xy=(p_gap_log, 0), xytext=(p_gap_log, label_y_high),
                ha='center', va='bottom', fontsize=8.5, color='dimgray',
                bbox=bbox_style, arrowprops=arrow_style)
    ax.annotate(f'P{p_popular_percentile}={int(popular_min)}',
                xy=(p_popular_log, 0), xytext=(p_popular_log, label_y_low),
                ha='center', va='bottom', fontsize=8.5, color='dimgray',
                bbox=bbox_style, arrowprops=arrow_style)

    ax.set_xticks([t for t in tick_vals if t <= arr_log.max() + 0.5])
    ax.set_xticklabels([tick_labels[i] for i, t in enumerate(tick_vals) if t <= arr_log.max() + 0.5])

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, y_max * 1.28)
    ax.set_xlim(arr_log.min(), arr_log.max())

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_path}")
    plt.close(fig)
    return os.path.basename(output_path)


log(f"| 指标 | 方案 | popular 边界 (P) | gap 下边界 (P) | gap 范围 | unpopular 上界 |")
log(f"| --- | --- | --- | --- | --- | --- |")

saved_plots = []

# 循环所有分组方案，每个生成 downloads + likes 两张图
for popular_pct, gap_pct, unpopular_pct in SCHEMES:
    scheme_suffix = f"{popular_pct}-{gap_pct}-{unpopular_pct}"

    dl_name = plot_distribution(
        values_all=all_downloads,
        xlabel='Number of Downloads (log scale)',
        title=f'Distribution of Downloads (Filtered, {total:,} models, scheme {scheme_suffix})',
        tick_vals=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        tick_labels=['1', '10', '100', '1K', '10K', '100K', '1M', '10M', '100M'],
        output_path=os.path.join(output_plots_dir, f'filtered_distribution_downloads_{scheme_suffix}.png'),
        popular_pct=popular_pct, gap_pct=gap_pct, unpopular_pct=unpopular_pct,
        metric_name='downloads',
    )
    saved_plots.append(dl_name)

    lk_name = plot_distribution(
        values_all=all_likes,
        xlabel='Number of Likes (log scale)',
        title=f'Distribution of Likes (Filtered, {total:,} models, scheme {scheme_suffix})',
        tick_vals=[0, 1, 2, 3, 4, 5],
        tick_labels=['1', '10', '100', '1K', '10K', '100K'],
        output_path=os.path.join(output_plots_dir, f'filtered_distribution_likes_{scheme_suffix}.png'),
        popular_pct=popular_pct, gap_pct=gap_pct, unpopular_pct=unpopular_pct,
        metric_name='likes',
    )
    saved_plots.append(lk_name)

log(f"")
log(f"## 生成的图")
log(f"")
for name in saved_plots:
    log(f"- ![{name}]({name})")

summary_path = os.path.join(output_plots_dir, 'visualization_summary.md')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(_log_lines) + '\n')
print(f"\n📝 汇总已写入: {summary_path}")
