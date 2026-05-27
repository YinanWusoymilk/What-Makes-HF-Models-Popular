import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --- 保存路径：跟脚本同目录 ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 加载数据 ---
print("正在加载元数据...")
with open(os.path.join(script_dir, '../data-collection/metadata-collection/2026-02-28_all_huggingface_models_2669509.json'), 'r') as f:
    models = json.load(f)

# 按 id 去重，与下游 metadata_features.py 保持一致
raw_count = len(models)
seen_ids = set()
unique_models = []
for m in models:
    mid = m.get('id', '')
    if mid not in seen_ids:
        seen_ids.add(mid)
        unique_models.append(m)
dup_count = raw_count - len(unique_models)
print(f"原始: {raw_count} 条，去重后: {len(unique_models)} 条（移除 {dup_count} 条重复 id）")
models = unique_models

all_downloads = [m.get('downloads', 0) for m in models]
all_likes = [m.get('likes', 0) for m in models]

total = len(models)
dl_nonzero = [d for d in all_downloads if d > 0]
lk_nonzero = [l for l in all_likes if l > 0]

print(f"总模型数: {total}")
print(f"Downloads > 0: {len(dl_nonzero)} ({len(dl_nonzero)/total*100:.1f}%)")
print(f"Likes > 0:     {len(lk_nonzero)} ({len(lk_nonzero)/total*100:.1f}%)")
print(f"Downloads - median(非零): {np.median(dl_nonzero):.0f}, mean(非零): {np.mean(dl_nonzero):.0f}, max: {max(all_downloads)}")
print(f"Likes     - median(非零): {np.median(lk_nonzero):.0f}, mean(非零): {np.mean(lk_nonzero):.0f}, max: {max(all_likes)}")

# === 图1: Downloads 分布 ===
fig1, ax1 = plt.subplots(figsize=(8, 5))

# 对非零数据取 log10，用 KDE 画密度
dl_log = np.log10(np.array(dl_nonzero, dtype=float))
dl_kde = gaussian_kde(dl_log, bw_method=0.2)
dl_x = np.linspace(dl_log.min(), dl_log.max(), 1000)
dl_y = dl_kde(dl_x)

ax1.plot(dl_x, dl_y, 'k-', linewidth=1.5)
ax1.fill_between(dl_x, dl_y, alpha=0.1, color='black')

# 百分位标注（在非零数据中）
dl_p50 = np.log10(np.percentile(dl_nonzero, 50))
dl_p90 = np.log10(np.percentile(dl_nonzero, 90))
ax1.axvline(dl_p50, color='black', linestyle='--', linewidth=1, label=f'Median: {np.percentile(dl_nonzero, 50):.0f}')
ax1.axvline(dl_p90, color='black', linestyle=':', linewidth=1, label=f'90th pctl: {np.percentile(dl_nonzero, 90):.0f}')

# x 轴标签转换为实际数值
tick_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8]
tick_labels = ['1', '10', '100', '1K', '10K', '100K', '1M', '10M', '100M']
ax1.set_xticks([t for t in tick_vals if t <= dl_log.max() + 0.5])
ax1.set_xticklabels([tick_labels[i] for i, t in enumerate(tick_vals) if t <= dl_log.max() + 0.5])

ax1.set_xlabel('Number of Downloads (log scale)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(f'Distribution of Downloads\n({len(dl_nonzero):,} models with downloads > 0 out of {total:,} total, deduplicated)', fontsize=13)
ax1.legend(fontsize=10)
ax1.set_ylim(bottom=0)

plt.tight_layout()
fig1.savefig(os.path.join(script_dir, 'raw_distribution_downloads.png'), dpi=300, bbox_inches='tight')
print("✅ 已保存: raw_distribution_downloads.png")
plt.close(fig1)

# === 图2: Likes 分布 ===
fig2, ax2 = plt.subplots(figsize=(8, 5))

lk_log = np.log10(np.array(lk_nonzero, dtype=float))
lk_kde = gaussian_kde(lk_log, bw_method=0.2)
lk_x = np.linspace(lk_log.min(), lk_log.max(), 1000)
lk_y = lk_kde(lk_x)

ax2.plot(lk_x, lk_y, 'k-', linewidth=1.5)
ax2.fill_between(lk_x, lk_y, alpha=0.1, color='black')

lk_p50 = np.log10(np.percentile(lk_nonzero, 50))
lk_p90 = np.log10(np.percentile(lk_nonzero, 90))
ax2.axvline(lk_p50, color='black', linestyle='--', linewidth=1, label=f'Median: {np.percentile(lk_nonzero, 50):.0f}')
ax2.axvline(lk_p90, color='black', linestyle=':', linewidth=1, label=f'90th pctl: {np.percentile(lk_nonzero, 90):.0f}')

tick_vals_lk = [0, 1, 2, 3, 4, 5]
tick_labels_lk = ['1', '10', '100', '1K', '10K', '100K']
ax2.set_xticks([t for t in tick_vals_lk if t <= lk_log.max() + 0.5])
ax2.set_xticklabels([tick_labels_lk[i] for i, t in enumerate(tick_vals_lk) if t <= lk_log.max() + 0.5])

ax2.set_xlabel('Number of Likes (log scale)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(f'Distribution of Likes\n({len(lk_nonzero):,} models with likes > 0 out of {total:,} total, deduplicated)', fontsize=13)
ax2.legend(fontsize=10)
ax2.set_ylim(bottom=0)

plt.tight_layout()
fig2.savefig(os.path.join(script_dir, 'raw_distribution_likes.png'), dpi=300, bbox_inches='tight')
print("✅ 已保存: raw_distribution_likes.png")
plt.close(fig2)