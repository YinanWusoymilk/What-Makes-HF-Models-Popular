import pandas as pd
import numpy as np
import os

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')

THRESHOLD = 0.7  # Spearman correlation threshold

# Full 31-feature set before correlation filtering.
all_features = [
    # Documentation (14)
    'word_count_yaml', 'word_count_content', 'has_video', 'num_code_blk', 'num_inline_code',
    'num_static_img', 'num_animated_img', 'num_lists', 'num_table', 'num_github_links',
    'num_huggingface_links', 'num_arxiv', 'has_bibtex', 'has_license',
    # Models (10)
    'has_config', 'has_model_index_result', 'has_dataset', 'num_dataset',
    'match_huggingface_dataset', 'model_size_bytes',
    'num_root_file', 'num_modules', 'num_model_files', 'has_quantized',
    # Platforms (3)
    'has_space', 'has_safetensors', 'has_widgetData',
    # Techniques (4)
    'has_pipeline_name', 'has_primary_implementation_library_name',
    'has_supported_additional_frameworks_Libraries', 'if_supported_libraries',
]


def run_correlation_analysis(sort_by):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n{'='*60}")
    print(f"Spearman correlation analysis: sorted by {sort_by}")
    print(f"{'='*60}")

    data_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)

    # Keep only the popular and unpopular groups; the gap buffer is excluded.
    data = data[data['category'].isin(['popular', 'unpopular'])]
    print(f"Sample size: {len(data)} models")

    X = data[all_features]

    # Compute the Spearman correlation matrix.
    corr_matrix = X.corr(method='spearman')

    # Save the correlation heatmap.
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=True, yticklabels=True, linewidths=0.5)
    plt.title(f'Spearman Correlation Heatmap (by {sort_by})', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(script_dir, f'spearman_heatmap_{sort_by}_{SCHEME}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

    # Find highly correlated feature pairs.
    print(f"\nHighly correlated feature pairs (|Spearman| > {THRESHOLD}):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > THRESHOLD:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                high_corr_pairs.append((feat1, feat2, corr_val))
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")

    if not high_corr_pairs:
        print("  No highly correlated feature pairs found.")

    # For each highly correlated pair, remove the feature with lower absolute
    # correlation to the binary popularity label.
    y = (data['category'] == 'popular').astype(int)
    target_corr = X.corrwith(y, method='spearman').abs()

    features_to_remove = set()
    for feat1, feat2, corr_val in high_corr_pairs:
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
        if target_corr[feat1] >= target_corr[feat2]:
            features_to_remove.add(feat2)
            print(f"  -> Remove {feat2} "
                  f"(target correlation {target_corr[feat2]:.3f} < {target_corr[feat1]:.3f})")
        else:
            features_to_remove.add(feat1)
            print(f"  -> Remove {feat1} "
                  f"(target correlation {target_corr[feat1]:.3f} < {target_corr[feat2]:.3f})")

    remaining_features = [f for f in all_features if f not in features_to_remove]
    print(f"\nRemoved features: {sorted(features_to_remove)}")
    print(f"Remaining features: {len(remaining_features)}")

    # Save the retained feature list.
    output_path = os.path.join(script_dir, f'remaining_features_{sort_by}_{SCHEME}.csv')
    pd.DataFrame({'feature': remaining_features}).to_csv(output_path, index=False)
    print(f"Saved retained feature list: {output_path}")

    return remaining_features, features_to_remove


if __name__ == '__main__':
    for sort_by in ['downloads', 'likes']:
        run_correlation_analysis(sort_by)

    print(f"\n{'='*60}")
    print("Correlation analysis complete.")
    print(f"{'='*60}")
