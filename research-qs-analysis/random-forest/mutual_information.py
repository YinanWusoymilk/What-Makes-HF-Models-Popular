import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')


def run_mi_selection(sort_by):
    print(f"\n{'='*60}")
    print(f"Mutual information feature selection: sorted by {sort_by}")
    print(f"{'='*60}")

    # Load data.
    data_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)
    data = data[data['category'].isin(['popular', 'unpopular'])]

    # Load the retained features from Step 1.
    remaining_path = os.path.join(script_dir, f'remaining_features_{sort_by}_{SCHEME}.csv')
    remaining_features = pd.read_csv(remaining_path)['feature'].tolist()
    print(f"Input features: {len(remaining_features)}")

    X = data[remaining_features]
    y = (data['category'] == 'popular').astype(int)

    # --- Compute mutual information ---
    print("Computing mutual information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({
        'Feature': remaining_features,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False).reset_index(drop=True)

    print("\nMutual information ranking:")
    for i, row in mi_df.iterrows():
        print(f"  {i+1:2d}. {row['Feature']:<50s} MI={row['MI_Score']:.4f}")

    # Save MI ranking.
    mi_output = os.path.join(script_dir, f'mi_scores_{sort_by}_{SCHEME}.csv')
    mi_df.to_csv(mi_output, index=False)
    print(f"\nSaved MI ranking: {mi_output}")

    # --- Select k using Random Forest 5-fold CV AUC ---
    print("\nSelecting k with Random Forest 5-fold CV AUC...")
    k_values = list(range(5, len(remaining_features) + 1))
    auc_scores = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        top_k_features = mi_df['Feature'].head(k).tolist()
        X_k = data[top_k_features]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X_k, y, cv=cv, scoring='roc_auc')
        mean_auc = scores.mean()
        auc_scores.append(mean_auc)

        if k % 5 == 0 or k == len(remaining_features):
            print(f"  k={k:2d}: AUC={mean_auc:.4f}")

    # Select the k with the highest AUC.
    best_idx = np.argmax(auc_scores)
    best_k = k_values[best_idx]
    best_auc = auc_scores[best_idx]
    print(f"\nBest k = {best_k}, AUC = {best_auc:.4f}")

    # Plot k vs AUC.
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, auc_scores, 'b-o', markersize=4)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k} (AUC={best_auc:.4f})')
    plt.xlabel('Number of Features (k)', fontsize=12)
    plt.ylabel('Mean AUC (5-fold CV)', fontsize=12)
    plt.title(f'Mutual Information Feature Selection (by {sort_by})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(script_dir, f'mi_optimal_k_{sort_by}_{SCHEME}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved k vs AUC plot: {plot_path}")

    # Save selected features.
    best_features = mi_df['Feature'].head(best_k).tolist()
    best_output = os.path.join(script_dir, f'selected_features_{sort_by}_{SCHEME}.csv')
    pd.DataFrame({
        'feature': best_features,
        'MI_Score': mi_df['MI_Score'].head(best_k).tolist()
    }).to_csv(best_output, index=False)
    print(f"Saved selected feature list: {best_output} ({best_k} features)")

    # Save k vs AUC results.
    k_auc_df = pd.DataFrame({'k': k_values, 'mean_auc': auc_scores})
    k_auc_df.to_csv(os.path.join(script_dir, f'k_vs_auc_{sort_by}_{SCHEME}.csv'), index=False)

    return best_features


if __name__ == '__main__':
    for sort_by in ['downloads', 'likes']:
        run_mi_selection(sort_by)

    print(f"\n{'='*60}")
    print("Mutual information feature selection complete.")
    print(f"{'='*60}")
