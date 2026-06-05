"""
Step 1+2: 特征选择 (SCHEME = 10-10-80, 已去掉 downloads top 1%)

- Step 1: Spearman 相关性, 删掉 |ρ| > 0.7 的特征对中与 target 相关性较弱者
- Step 2: Mutual Information 排名 + 用 RF 5-fold CV AUC 找最优 k

跟 ../../research-qs-analysis/random-forest/{correlation_detection.py, mutual_information.py} 逻辑完全一致, 但跑在
robust_prep.py 产出的 cleaned data (去掉 downloads top 1% 后) 上, 输出本目录的
selected_features_<sort_by>_10-10-80.csv 供 robust_analysis.py 使用。
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
THRESHOLD = 0.7

# 31 个原始特征
all_features = [
    # Documentation
    'word_count_yaml', 'word_count_content', 'has_video', 'num_code_blk', 'num_inline_code',
    'num_static_img', 'num_animated_img', 'num_lists', 'num_table', 'num_github_links',
    'num_huggingface_links', 'num_arxiv', 'has_bibtex', 'has_license',
    # Models (10)
    'has_config', 'has_model_index_result', 'has_dataset', 'num_dataset',
    'match_huggingface_dataset', 'model_size_bytes',
    'num_root_file', 'num_modules', 'num_model_files', 'has_quantized',
    # Platforms
    'has_space', 'has_safetensors', 'has_widgetData',
    # Techniques
    'has_pipeline_name', 'has_primary_implementation_library_name',
    'has_supported_additional_frameworks_Libraries', 'if_supported_libraries',
]


def run_correlation(sort_by):
    print(f"\n--- Spearman Correlation (by {sort_by}) ---")
    data_path = os.path.join(script_dir, f'robust_filtered_by_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)
    data = data[data['category'].isin(['popular', 'unpopular'])]
    print(f"  数据量: {len(data)} 条")

    X = data[all_features]
    corr_matrix = X.corr(method='spearman')

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=True, yticklabels=True, linewidths=0.5)
    plt.title(f'Spearman Correlation Heatmap (by {sort_by}, robust {SCHEME})', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(script_dir, f'spearman_heatmap_{sort_by}_{SCHEME}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 热力图保存: {heatmap_path}")

    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > THRESHOLD:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                        corr_matrix.iloc[i, j]))
    print(f"  高度相关特征对 (|ρ|>{THRESHOLD}): {len(high_corr_pairs)}")

    y = (data['category'] == 'popular').astype(int)
    target_corr = X.corrwith(y, method='spearman').abs()

    features_to_remove = set()
    for f1, f2, cv in high_corr_pairs:
        if f1 in features_to_remove or f2 in features_to_remove:
            continue
        if target_corr[f1] >= target_corr[f2]:
            features_to_remove.add(f2)
            print(f"    {f1} <-> {f2} (ρ={cv:.3f}) → 删 {f2}")
        else:
            features_to_remove.add(f1)
            print(f"    {f1} <-> {f2} (ρ={cv:.3f}) → 删 {f1}")

    remaining = [f for f in all_features if f not in features_to_remove]
    pd.DataFrame({'feature': remaining}).to_csv(
        os.path.join(script_dir, f'remaining_features_{sort_by}_{SCHEME}.csv'), index=False)
    print(f"  剩余特征: {len(remaining)}")
    return remaining


def run_mi(sort_by, remaining_features):
    print(f"\n--- Mutual Information (by {sort_by}) ---")
    data_path = os.path.join(script_dir, f'robust_filtered_by_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)
    data = data[data['category'].isin(['popular', 'unpopular'])]

    X = data[remaining_features]
    y = (data['category'] == 'popular').astype(int)

    print(f"  计算 MI...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'Feature': remaining_features, 'MI_Score': mi_scores})\
        .sort_values('MI_Score', ascending=False).reset_index(drop=True)
    mi_df.to_csv(os.path.join(script_dir, f'mi_scores_{sort_by}_{SCHEME}.csv'), index=False)
    print(f"  Top-5 MI: {mi_df.head(5)['Feature'].tolist()}")

    print(f"  扫 k=5..{len(remaining_features)} 用 RF + 5-fold CV AUC 找最优 k...")
    k_values = list(range(5, len(remaining_features) + 1))
    auc_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k in k_values:
        top_k = mi_df['Feature'].head(k).tolist()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, data[top_k], y, cv=cv, scoring='roc_auc')
        auc_scores.append(scores.mean())
        if k % 5 == 0 or k == len(remaining_features):
            print(f"    k={k:2d}: AUC={scores.mean():.4f}")

    best_idx = int(np.argmax(auc_scores))
    best_k = k_values[best_idx]
    best_auc = auc_scores[best_idx]
    print(f"  最优 k = {best_k} (AUC = {best_auc:.4f})")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, auc_scores, 'b-o', markersize=4)
    plt.axvline(x=best_k, color='r', linestyle='--',
                label=f'Best k={best_k} (AUC={best_auc:.4f})')
    plt.xlabel('Number of Features (k)')
    plt.ylabel('Mean AUC (5-fold CV)')
    plt.title(f'MI Feature Selection (by {sort_by}, robust {SCHEME})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'mi_optimal_k_{sort_by}_{SCHEME}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    selected = mi_df['Feature'].head(best_k).tolist()
    pd.DataFrame({'feature': selected,
                  'MI_Score': mi_df['MI_Score'].head(best_k).tolist()}).to_csv(
        os.path.join(script_dir, f'selected_features_{sort_by}_{SCHEME}.csv'), index=False)

    pd.DataFrame({'k': k_values, 'mean_auc': auc_scores}).to_csv(
        os.path.join(script_dir, f'k_vs_auc_{sort_by}_{SCHEME}.csv'), index=False)
    return selected


def main():
    print(f"=== Feature selection (robust, SCHEME = {SCHEME}, 去掉 downloads top 1% 后) ===")
    for sort_by in ['downloads', 'likes']:
        remaining = run_correlation(sort_by)
        run_mi(sort_by, remaining)
    print(f"\n=== ✅ Feature selection done ===")


if __name__ == '__main__':
    main()
