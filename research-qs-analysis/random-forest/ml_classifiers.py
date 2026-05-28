import pandas as pd
import numpy as np
import os

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')

# Evaluation metrics used in the paper: primary = roc_auc; secondary =
# precision, recall, f1, and accuracy.
SCORING = {
    'auc': 'roc_auc',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
}
ACCEPTABLE_AUC = 0.7
PERM_REPEATS = 30


def get_models_and_params():
    """Five classifiers and their hyperparameter grids.

    LinearSVC is used instead of SVC(kernel='linear', probability=True) because
    it is substantially faster on this dataset. The ROC AUC scorer uses the
    decision_function output automatically.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    models = {
        'RandomForest': {
            'pipeline': Pipeline([
                ('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
            ]),
            'params': {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [None, 10, 20],
            }
        },
        'DecisionTree': {
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
            ]),
            'params': {
                'clf__max_depth': [5, 10, 20, None],
                'clf__min_samples_split': [2, 5, 10],
            }
        },
        'SVM': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LinearSVC(random_state=42, dual='auto', max_iter=5000, class_weight='balanced'))
            ]),
            'params': {
                'clf__C': [0.01, 0.1, 1, 10],
            }
        },
        'NaiveBayes': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ]),
            'params': {
                'clf__var_smoothing': [1e-9, 1e-8, 1e-7],
            }
        },
        'KNN': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(n_jobs=-1))
            ]),
            'params': {
                'clf__n_neighbors': [3, 5, 7, 11],
                'clf__weights': ['uniform', 'distance'],
            }
        },
    }
    return models


def extract_metric(cv_results, best_index, metric):
    mean = cv_results[f'mean_test_{metric}'][best_index]
    std = cv_results[f'std_test_{metric}'][best_index]
    return float(mean), float(std)


def run_classifiers(sort_by):
    from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
    from sklearn.inspection import permutation_importance

    print(f"\n{'='*60}")
    print(f"ML classifiers: sorted by {sort_by}")
    print(f"{'='*60}")

    data_path = os.path.join(FILTERED_DIR, f'filtered_model_data_by_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)
    data = data[data['category'].isin(['popular', 'unpopular'])]

    selected_path = os.path.join(script_dir, f'selected_features_{sort_by}_{SCHEME}.csv')
    selected_features = pd.read_csv(selected_path)['feature'].tolist()
    print(f"Number of selected features: {len(selected_features)}")

    X = data[selected_features].values
    y = (data['category'] == 'popular').astype(int).values
    print(f"Sample size: {len(X)} models "
          f"(popular={int(y.sum())}, unpopular={int(len(y)-y.sum())})")

    # 10-fold x 10 repeats = 100 folds, with multiple scoring metrics.
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    models = get_models_and_params()

    all_results = []
    fitted_estimators = {}  # Store each model's best estimator for later use.

    for model_name, model_config in models.items():
        print(f"\n--- {model_name} ---")
        print("Running GridSearchCV with multiple metrics...")

        grid_search = GridSearchCV(
            estimator=model_config['pipeline'],
            param_grid=model_config['params'],
            cv=cv,
            scoring=SCORING,
            refit='auc',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X, y)

        bi = grid_search.best_index_
        auc_m, auc_s = extract_metric(grid_search.cv_results_, bi, 'auc')
        acc_m, acc_s = extract_metric(grid_search.cv_results_, bi, 'accuracy')
        pre_m, pre_s = extract_metric(grid_search.cv_results_, bi, 'precision')
        rec_m, rec_s = extract_metric(grid_search.cv_results_, bi, 'recall')
        f1_m, f1_s = extract_metric(grid_search.cv_results_, bi, 'f1')

        print(f"Best params: {grid_search.best_params_}")
        print(f"  AUC       : {auc_m:.4f} ± {auc_s:.4f}")
        print(f"  Accuracy  : {acc_m:.4f} ± {acc_s:.4f}")
        print(f"  Precision : {pre_m:.4f} ± {pre_s:.4f}")
        print(f"  Recall    : {rec_m:.4f} ± {rec_s:.4f}")
        print(f"  F1        : {f1_m:.4f} ± {f1_s:.4f}")

        all_results.append({
            'Model': model_name,
            'AUC_mean': auc_m, 'AUC_std': auc_s,
            'Accuracy_mean': acc_m, 'Accuracy_std': acc_s,
            'Precision_mean': pre_m, 'Precision_std': pre_s,
            'Recall_mean': rec_m, 'Recall_std': rec_s,
            'F1_mean': f1_m, 'F1_std': f1_s,
            'Acceptable_AUC>0.7': auc_m > ACCEPTABLE_AUC,
            'Best_Params': str(grid_search.best_params_),
        })
        fitted_estimators[model_name] = grid_search.best_estimator_

        # Save Random Forest impurity-based importance as a reference output.
        if model_name == 'RandomForest':
            rf_clf = grid_search.best_estimator_.named_steps['clf']
            imp_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_clf.feature_importances_,
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            imp_path = os.path.join(script_dir, f'rf_impurity_importance_{sort_by}_{SCHEME}.csv')
            imp_df.to_csv(imp_path, index=False)
            print(f"Saved RF impurity importance reference: {imp_path}")

    results_df = pd.DataFrame(all_results).sort_values('AUC_mean', ascending=False).reset_index(drop=True)
    results_path = os.path.join(script_dir, f'ml_classifier_results_{sort_by}_{SCHEME}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved classifier comparison results: {results_path}")

    # --- Select the best model and run permutation importance ---
    best_model_name = results_df.iloc[0]['Model']
    best_auc = results_df.iloc[0]['AUC_mean']
    print(f"\n📊 Best model: {best_model_name} (AUC={best_auc:.4f}) "
          f"{'✓ acceptable (>0.7)' if best_auc > ACCEPTABLE_AUC else '⚠️ AUC ≤ 0.7, performance unacceptable per paper threshold'}")

    if best_auc <= ACCEPTABLE_AUC:
        print("Skipping permutation importance because the best model does not meet the acceptable AUC threshold.")
        return results_df

    print(f"\n--- Permutation Importance on {best_model_name} (n_repeats={PERM_REPEATS}) ---")
    # Compute permutation importance on a holdout test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Refit the best model on the training split before computing test-set
    # permutation importance.
    best_est = fitted_estimators[best_model_name]
    from sklearn.base import clone
    best_est_train = clone(best_est)
    best_est_train.fit(X_train, y_train)

    perm = permutation_importance(
        best_est_train, X_test, y_test,
        n_repeats=PERM_REPEATS, random_state=42,
        scoring='roc_auc', n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        'Feature': selected_features,
        'PermImportance_mean': perm.importances_mean,
        'PermImportance_std': perm.importances_std,
    }).sort_values('PermImportance_mean', ascending=False).reset_index(drop=True)

    perm_path = os.path.join(script_dir, f'permutation_importance_{sort_by}_{SCHEME}.csv')
    perm_df.to_csv(perm_path, index=False)
    print(f"Saved permutation importance: {perm_path}")
    print("\nTop 10:")
    for _, r in perm_df.head(10).iterrows():
        print(f"  {r['Feature']:<50s} {r['PermImportance_mean']:+.4f} ± {r['PermImportance_std']:.4f}")

    # Save best-model metadata for the generated Markdown summary.
    pd.DataFrame([{
        'sort_by': sort_by,
        'best_model': best_model_name,
        'best_auc': best_auc,
        'best_params': results_df.iloc[0]['Best_Params'],
    }]).to_csv(os.path.join(script_dir, f'best_model_{sort_by}_{SCHEME}.csv'), index=False)

    return results_df


# -------------------- Markdown summary --------------------

def _read_dropped(sort_by):
    from correlation_detection import all_features
    remaining = pd.read_csv(os.path.join(script_dir, f'remaining_features_{sort_by}_{SCHEME}.csv'))['feature'].tolist()
    dropped = [f for f in all_features if f not in remaining]
    return dropped, remaining


def _read_mi(sort_by):
    sel = pd.read_csv(os.path.join(script_dir, f'selected_features_{sort_by}_{SCHEME}.csv'))
    k_auc = pd.read_csv(os.path.join(script_dir, f'k_vs_auc_{sort_by}_{SCHEME}.csv'))
    best_row = k_auc.loc[k_auc['mean_auc'].idxmax()]
    return sel, int(best_row['k']), float(best_row['mean_auc'])


def _read_results(sort_by):
    return pd.read_csv(os.path.join(script_dir, f'ml_classifier_results_{sort_by}_{SCHEME}.csv'))


def _read_perm(sort_by):
    return pd.read_csv(os.path.join(script_dir, f'permutation_importance_{sort_by}_{SCHEME}.csv'))


def _read_best_model(sort_by):
    return pd.read_csv(os.path.join(script_dir, f'best_model_{sort_by}_{SCHEME}.csv')).iloc[0]


def generate_md_summary():
    md = []
    md.append(f"# RQ2: Feature Importance Summary (scheme {SCHEME})")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append("- **Step 1 - Correlation filtering**: Compute Spearman correlation across the 31 features. For every pair with |ρ| > 0.7, retain the feature with the higher absolute correlation to the binary popularity label and remove the other feature.")
    md.append("- **Step 2 - Mutual information ranking**: Compute mutual information for the retained features, sweep k=5..n, and select k using Random Forest 5-fold cross-validation AUC.")
    md.append(f"- **Step 3 - Classifier evaluation**: Compare five classifiers (Random Forest / Decision Tree / Linear SVM / Gaussian Naive Bayes / KNN) using GridSearchCV with **10-fold x 10 repeats = 100 folds**. ROC AUC is the primary metric; accuracy, precision, recall, and F1 are secondary metrics. AUC > {ACCEPTABLE_AUC} is treated as the reference threshold for acceptable discrimination.")
    md.append(f"- **Step 4 - Permutation importance**: For the best model by AUC, compute permutation importance on a stratified 80/20 holdout test set with `permutation_importance(n_repeats={PERM_REPEATS}, scoring='roc_auc')`.")
    md.append("")

    for sort_by in ['downloads', 'likes']:
        try:
            dropped, remaining = _read_dropped(sort_by)
            sel_df, best_k, best_k_auc = _read_mi(sort_by)
            results_df = _read_results(sort_by)
        except FileNotFoundError as e:
            md.append(f"## Sorted by {sort_by}")
            md.append("")
            md.append(f"Missing intermediate output: {e}")
            md.append("")
            continue

        md.append(f"## Sorted by {sort_by}")
        md.append("")

        md.append(f"### Step 1: Correlation filtering")
        md.append("")
        md.append(f"Removed {len(dropped)} features and retained **{len(remaining)}** features.")
        if dropped:
            md.append(f"Removed features: {', '.join(f'`{f}`' for f in dropped)}")
        else:
            md.append("No highly correlated feature pairs were found.")
        md.append("")

        md.append(f"### Step 2: Mutual information ranking")
        md.append("")
        md.append(f"Optimal **k = {best_k}** (Random Forest 5-fold CV AUC = {best_k_auc:.4f})")
        md.append("")
        md.append("| Rank | Feature | MI Score |")
        md.append("| --- | --- | --- |")
        for i, r in sel_df.iterrows():
            md.append(f"| {i+1} | `{r['feature']}` | {r['MI_Score']:.4f} |")
        md.append("")

        md.append(f"### Step 3: Classifier ranking by AUC (primary = AUC, secondary = Acc/P/R/F1)")
        md.append("")
        md.append("| Model | AUC | Accuracy | Precision | Recall | F1 | Acceptable (AUC>0.7) |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, r in results_df.iterrows():
            md.append(
                f"| {r['Model']} | {r['AUC_mean']:.4f} ± {r['AUC_std']:.4f} | "
                f"{r['Accuracy_mean']:.4f} ± {r['Accuracy_std']:.4f} | "
                f"{r['Precision_mean']:.4f} ± {r['Precision_std']:.4f} | "
                f"{r['Recall_mean']:.4f} ± {r['Recall_std']:.4f} | "
                f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f} | "
                f"{'✓' if r['Acceptable_AUC>0.7'] else '✗'} |"
            )
        md.append("")

        # Permutation importance on the best model.
        try:
            best_info = _read_best_model(sort_by)
            perm_df = _read_perm(sort_by)
            md.append(f"### Step 4: Permutation importance on best model = **{best_info['best_model']}** (AUC={best_info['best_auc']:.4f})")
            md.append("")
            md.append(f"Best params: `{best_info['best_params']}`")
            md.append("")
            md.append("| Rank | Feature | PermImportance (Δ AUC) |")
            md.append("| --- | --- | --- |")
            for i, r in perm_df.iterrows():
                md.append(f"| {i+1} | `{r['Feature']}` | {r['PermImportance_mean']:+.4f} ± {r['PermImportance_std']:.4f} |")
            md.append("")
        except FileNotFoundError:
            md.append("### Step 4: Permutation importance")
            md.append("")
            md.append("Permutation importance was skipped because the best model did not meet the acceptable AUC threshold.")
            md.append("")

    # Cross-indicator comparison.
    try:
        dl_dropped, _ = _read_dropped('downloads')
        lk_dropped, _ = _read_dropped('likes')
        dl_sel, _, _ = _read_mi('downloads')
        lk_sel, _, _ = _read_mi('likes')
        md.append("## Cross-indicator comparison (downloads vs likes)")
        md.append("")
        md.append(f"- **Correlation filtering**: downloads removed {sorted(dl_dropped)}; likes removed {sorted(lk_dropped)}")
        md.append(f"- **Removed by both indicators**: {sorted(set(dl_dropped) & set(lk_dropped))}")
        md.append(f"- **Selected by MI under both indicators**: {sorted(set(dl_sel['feature']) & set(lk_sel['feature']))}")
        try:
            dl_perm = set(_read_perm('downloads').head(10)['Feature'])
            lk_perm = set(_read_perm('likes').head(10)['Feature'])
            md.append(f"- **Permutation-importance top-10 overlap**: {sorted(dl_perm & lk_perm)}")
        except FileNotFoundError:
            pass
        md.append("")
    except FileNotFoundError:
        pass

    summary_path = os.path.join(script_dir, f'rq2_summary_{SCHEME}.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f"\nWrote summary: {summary_path}")


if __name__ == '__main__':
    for sort_by in ['downloads', 'likes']:
        run_classifiers(sort_by)

    generate_md_summary()

    print(f"\n{'='*60}")
    print("All ML classifier analyses complete.")
    print(f"{'='*60}")
