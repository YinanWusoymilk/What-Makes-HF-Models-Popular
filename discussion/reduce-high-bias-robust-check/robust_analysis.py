import pandas as pd
import numpy as np
import ast
import os
from scipy.stats import mannwhitneyu, chi2_contingency
from cliffs_delta import cliffs_delta
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'

# 特征列表（跟主分析一致）
features_continuous = [
    'num_arxiv', 'num_dataset', 'num_model_files', 'model_size_bytes',
    'num_root_file', 'num_modules', 'num_lists', 'num_static_img', 'num_animated_img',
    'num_github_links', 'num_huggingface_links', 'num_code_blk', 'num_inline_code',
    'num_table', 'word_count_yaml', 'word_count_content'
]
features_binary = [
    'has_dataset', 'match_huggingface_dataset', 'has_safetensors', 'has_quantized',
    'has_license', 'has_space', 'has_pipeline_name',
    'has_primary_implementation_library_name', 'if_supported_libraries',
    'has_supported_additional_frameworks_Libraries', 'has_widgetData', 'has_config',
    'has_model_index_result', 'has_video', 'has_bibtex'
]

MIN_SAMPLES = 30          # 跟主 RQ3 一致
BOOTSTRAP_N = 1000        # bootstrap CI 迭代数
RNG_SEED = 42
PERM_REPEATS = 30         # permutation importance 重复次数 (跟 baseline ml_classifiers.py 对齐)
ACCEPTABLE_AUC = 0.7      # AUC > 0.7 才跑 perm importance (跟 baseline 对齐)


# ============================================================
# RQ1: Significance Analysis
# ============================================================

def compute_cramers_v(table):
    stat, p, dof, expected = chi2_contingency(table)
    n = table.sum().sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(stat / (n * min_dim)))


def interpret_cliffs_delta(d):
    abs_d = abs(d)
    if abs_d < 0.147: return 'negligible'
    elif abs_d < 0.33: return 'small'
    elif abs_d < 0.474: return 'medium'
    else: return 'large'


def interpret_cramers_v(v):
    if v < 0.1: return 'negligible'
    elif v < 0.3: return 'small'
    elif v < 0.5: return 'medium'
    else: return 'large'


def run_rq1(data, sort_by, output_dir):
    print(f"\n  [RQ1] 显著性分析 (by {sort_by})...")
    popular = data[data['category'] == 'popular']
    unpopular = data[data['category'] == 'unpopular']
    subset = data[data['category'].isin(['popular', 'unpopular'])]

    results = []
    for feature in features_continuous:
        if feature not in data.columns:
            continue
        x1 = popular[feature].dropna()
        x2 = unpopular[feature].dropna()
        if len(x1) == 0 or len(x2) == 0:
            continue
        stat, p_value = mannwhitneyu(x1, x2, alternative='two-sided')
        d, _ = cliffs_delta(x1, x2)
        results.append({
            'Feature': feature, 'Test_Type': 'Continuous', 'Test_Applied': 'Mann-Whitney U',
            'p_value': p_value, 'Effect_Size_Measure': "Cliff's Delta",
            'Effect_Size_Value': d, 'Effect_Size_Interpretation': interpret_cliffs_delta(d),
        })

    for feature in features_binary:
        if feature not in data.columns:
            continue
        table = pd.crosstab(index=subset[feature], columns=subset['category'])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        stat, p_value, dof, expected = chi2_contingency(table)
        v = compute_cramers_v(table)
        results.append({
            'Feature': feature, 'Test_Type': 'Binary', 'Test_Applied': 'Chi-square',
            'p_value': p_value, 'Effect_Size_Measure': "Cramér's V",
            'Effect_Size_Value': v, 'Effect_Size_Interpretation': interpret_cramers_v(v),
        })

    results_df = pd.DataFrame(results)
    reject_fdr, pvals_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh', alpha=0.05)
    results_df['p_value_fdr_corrected'] = pvals_fdr
    results_df['significant_after_fdr'] = reject_fdr

    path = os.path.join(output_dir, f'robust_rq1_significance_{sort_by}_{SCHEME}.csv')
    results_df.to_csv(path, index=False)
    sig = int(results_df['significant_after_fdr'].sum())
    print(f"  [RQ1] 显著特征: {sig}/{len(results_df)}")
    print(f"  [RQ1] 效应量: {results_df['Effect_Size_Interpretation'].value_counts().to_dict()}")
    return results_df


# ============================================================
# RQ2: ML Classifiers (复用原 RQ2 选好的特征)
# ============================================================

def get_models():
    return {
        'RandomForest': Pipeline([('clf', RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))]),
        'DecisionTree': Pipeline([('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced'))]),
        'SVM': Pipeline([('scaler', StandardScaler()), ('clf', LinearSVC(random_state=42, dual='auto', max_iter=5000, class_weight='balanced'))]),
        'NaiveBayes': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
        'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_jobs=-1))]),
    }


def get_param_grids():
    return {
        'RandomForest': {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 10, 20]},
        'DecisionTree': {'clf__max_depth': [5, 10, 20, None], 'clf__min_samples_split': [2, 5, 10]},
        'SVM': {'clf__C': [0.01, 0.1, 1, 10]},
        'NaiveBayes': {'clf__var_smoothing': [1e-9, 1e-8, 1e-7]},
        'KNN': {'clf__n_neighbors': [3, 5, 7, 11], 'clf__weights': ['uniform', 'distance']},
    }


def run_rq2(data, sort_by, output_dir):
    """跑 RQ2,返回 (results_df, rf_best_params_dict) 供 RQ3 复用 RF 超参"""
    print(f"\n  [RQ2] ML 分类器 (by {sort_by})...")

    sel_path = os.path.join(script_dir, f'selected_features_{sort_by}_{SCHEME}.csv')
    selected_features = pd.read_csv(sel_path)['feature'].tolist()
    print(f"  使用 {len(selected_features)} 个特征 (本目录 feature_selection.py 重新选出)")

    X = data[selected_features].values
    y = (data['category'] == 'popular').astype(int).values

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    models = get_models()
    param_grids = get_param_grids()

    all_results = []
    rf_best_params = None
    rf_best_estimator = None
    for model_name, pipeline in models.items():
        print(f"\n    --- {model_name} ---")
        print(f"    正在进行 GridSearch (cv=10×10, scoring=roc_auc)...", flush=True)
        grid = GridSearchCV(pipeline, param_grids[model_name], cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X, y)
        auc = grid.best_score_
        std = grid.cv_results_['std_test_score'][grid.best_index_]
        print(f"    {model_name}: AUC={auc:.4f} ± {std:.4f}  best={grid.best_params_}", flush=True)
        all_results.append({'Model': model_name, 'AUC_mean': auc, 'AUC_std': std,
                            'Best_Params': str(grid.best_params_)})
        if model_name == 'RandomForest':
            # 去掉 'clf__' 前缀,直接喂给 RandomForestClassifier(**)
            rf_best_params = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}
            rf_best_estimator = grid.best_estimator_

            # RF impurity-based importance (跟 baseline ml_classifiers.py:158-167 一致)
            rf_clf = grid.best_estimator_.named_steps['clf']
            imp_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': rf_clf.feature_importances_,
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            imp_path = os.path.join(output_dir, f'robust_rf_impurity_importance_{sort_by}_{SCHEME}.csv')
            imp_df.to_csv(imp_path, index=False)
            print(f"    ✅ RF impurity importance 已保存: {imp_path}")

    results_df = pd.DataFrame(all_results).sort_values('AUC_mean', ascending=False).reset_index(drop=True)
    path = os.path.join(output_dir, f'robust_rq2_classifiers_{sort_by}_{SCHEME}.csv')
    results_df.to_csv(path, index=False)
    best_model_name = results_df.iloc[0]['Model']
    best_auc = results_df.iloc[0]['AUC_mean']
    print(f"  [RQ2] Best: {best_model_name} (AUC={best_auc:.4f})")

    # 保存 best_model 信息 (跟 baseline ml_classifiers.py:216-221 对齐)
    pd.DataFrame([{
        'sort_by': sort_by,
        'best_model': best_model_name,
        'best_auc': best_auc,
        'best_params': results_df.iloc[0]['Best_Params'],
    }]).to_csv(os.path.join(output_dir, f'robust_best_model_{sort_by}_{SCHEME}.csv'), index=False)

    return results_df, rf_best_params, selected_features, rf_best_estimator, best_model_name, best_auc


# ============================================================
# Permutation Importance (跟 baseline ml_classifiers.py:174-213 对齐)
# ============================================================

def run_permutation_importance(data, sort_by, best_estimator, best_model_name, best_auc,
                                selected_features, output_dir):
    """跟 baseline 一样的 perm importance: 80/20 stratified holdout, refit, n_repeats=30."""
    if best_auc <= ACCEPTABLE_AUC:
        print(f"  ⚠️ 跳过 permutation importance, best AUC={best_auc:.4f} ≤ {ACCEPTABLE_AUC}")
        return None

    X = data[selected_features].values
    y = (data['category'] == 'popular').astype(int).values

    print(f"\n  [Perm Importance] on {best_model_name} (n_repeats={PERM_REPEATS})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG_SEED
    )

    # clone + refit on train (avoid using full-data refit estimator for OOS perm importance)
    best_est_train = clone(best_estimator)
    best_est_train.fit(X_train, y_train)

    perm = permutation_importance(
        best_est_train, X_test, y_test,
        n_repeats=PERM_REPEATS, random_state=RNG_SEED,
        scoring='roc_auc', n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        'Feature': selected_features,
        'PermImportance_mean': perm.importances_mean,
        'PermImportance_std': perm.importances_std,
    }).sort_values('PermImportance_mean', ascending=False).reset_index(drop=True)

    perm_path = os.path.join(output_dir, f'robust_permutation_importance_{sort_by}_{SCHEME}.csv')
    perm_df.to_csv(perm_path, index=False)
    print(f"  ✅ Permutation importance 已保存: {perm_path}")
    print("  排名前 10:")
    for _, r in perm_df.head(10).iterrows():
        print(f"    {r['Feature']:<50s} {r['PermImportance_mean']:+.4f} ± {r['PermImportance_std']:.4f}")
    return perm_df


# ============================================================
# RQ3: Cross-group (mirrors cross_group_rf_analysis.py)
# ============================================================

def make_rf(best_params):
    # class_weight='balanced' 对应 popular:unpopular ~1:8 不平衡, 跟 baseline 保持一致
    return RandomForestClassifier(random_state=RNG_SEED, n_jobs=-1, class_weight='balanced', **best_params)


def bootstrap_auc_ci(y_true, y_pred_prob, n_iter=BOOTSTRAP_N, rng_seed=RNG_SEED):
    """对 (y_true, y_pred_prob) bootstrap 重采样,返回 AUC mean / std / 2.5% / 97.5%"""
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred_prob[idx]))
    aucs = np.array(aucs)
    return aucs.mean(), aucs.std(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def get_valid_groups(data, group_col):
    valid = []
    for group in data[group_col].dropna().unique():
        g = data[data[group_col] == group]
        pop = (g['category'] == 'popular').sum()
        unpop = (g['category'] == 'unpopular').sum()
        if pop >= MIN_SAMPLES and unpop >= MIN_SAMPLES:
            valid.append(group)
        else:
            print(f"    跳过 {group} (popular={pop}, unpopular={unpop}, 不足 {MIN_SAMPLES})")
    return sorted(valid)


def run_within_group_baseline(data, group_col, valid_groups, features, best_params):
    rows = []
    print(f"  --- Within-group baseline (10-fold CV in same group) ---")
    for group in valid_groups:
        g = data[data[group_col] == group]
        X = g[features].values
        y = (g['category'] == 'popular').astype(int).values
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=RNG_SEED)
        aucs = []
        for tr_idx, te_idx in cv.split(X, y):
            rf = make_rf(best_params)
            rf.fit(X[tr_idx], y[tr_idx])
            prob = rf.predict_proba(X[te_idx])[:, 1]
            aucs.append(roc_auc_score(y[te_idx], prob))
        aucs = np.array(aucs)
        print(f"    {group}: AUC={aucs.mean():.4f} ± {aucs.std():.4f} (n={len(g)})")
        rows.append({'Group': group, 'N': len(g),
                     'AUC_mean': aucs.mean(), 'AUC_std': aucs.std()})
    return pd.DataFrame(rows)


def run_single_to_single(data, group_col, valid_groups, features, best_params):
    rows = []
    print(f"  --- 单对单 cross-group (train on A → test on B), bootstrap {BOOTSTRAP_N} CI ---")
    for tr in valid_groups:
        tr_data = data[data[group_col] == tr]
        X_tr = tr_data[features].values
        y_tr = (tr_data['category'] == 'popular').astype(int).values
        rf = make_rf(best_params)
        rf.fit(X_tr, y_tr)
        for te in valid_groups:
            if te == tr:
                continue
            te_data = data[data[group_col] == te]
            X_te = te_data[features].values
            y_te = (te_data['category'] == 'popular').astype(int).values
            prob = rf.predict_proba(X_te)[:, 1]
            mean_auc, std_auc, ci_lo, ci_hi = bootstrap_auc_ci(y_te, prob)
            print(f"    {tr} → {te}: AUC={mean_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
            rows.append({'Train': tr, 'Test': te,
                         'AUC_mean': mean_auc, 'AUC_std': std_auc,
                         'CI_lo': ci_lo, 'CI_hi': ci_hi,
                         'N_train': len(tr_data), 'N_test': len(te_data)})
    return pd.DataFrame(rows)


def run_combined_to_single(data, group_col, valid_groups, features, best_params):
    rows = []
    print(f"  --- 多对一 cross-group (others → target), bootstrap {BOOTSTRAP_N} CI ---")
    for te in valid_groups:
        tr_data = data[(data[group_col].isin(valid_groups)) & (data[group_col] != te)]
        te_data = data[data[group_col] == te]
        X_tr = tr_data[features].values
        y_tr = (tr_data['category'] == 'popular').astype(int).values
        X_te = te_data[features].values
        y_te = (te_data['category'] == 'popular').astype(int).values
        rf = make_rf(best_params)
        rf.fit(X_tr, y_tr)
        prob = rf.predict_proba(X_te)[:, 1]
        mean_auc, std_auc, ci_lo, ci_hi = bootstrap_auc_ci(y_te, prob)
        print(f"    others → {te}: AUC={mean_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        rows.append({'Train': 'others', 'Test': te,
                     'AUC_mean': mean_auc, 'AUC_std': std_auc,
                     'CI_lo': ci_lo, 'CI_hi': ci_hi,
                     'N_train': len(tr_data), 'N_test': len(te_data)})
    return pd.DataFrame(rows)


def make_auc_matrix(single_df, valid_groups):
    m = pd.DataFrame(index=valid_groups, columns=valid_groups, dtype=float)
    for _, r in single_df.iterrows():
        m.loc[r['Train'], r['Test']] = r['AUC_mean']
    for g in valid_groups:
        m.loc[g, g] = np.nan
    return m


def run_rq3_cross_group(data, sort_by, group_col, group_label, features, best_params, output_dir):
    print(f"\n  [RQ3] Cross-{group_label} (by {sort_by})...")
    if group_col not in data.columns:
        print(f"  ⚠️ 列 {group_col} 不存在,跳过")
        return None

    valid_groups = get_valid_groups(data, group_col)
    print(f"  有效 {group_label}: {len(valid_groups)} 个: {valid_groups}")
    if len(valid_groups) < 2:
        print(f"  ⚠️ 有效 group 不足 2 个,跳过")
        return None

    within_df = run_within_group_baseline(data, group_col, valid_groups, features, best_params)
    within_df.to_csv(os.path.join(output_dir, f'robust_rq3_within_{group_label}_{sort_by}_{SCHEME}.csv'), index=False)

    single_df = run_single_to_single(data, group_col, valid_groups, features, best_params)
    single_df.to_csv(os.path.join(output_dir, f'robust_rq3_cross_{group_label}_single_{sort_by}_{SCHEME}.csv'), index=False)

    matrix = make_auc_matrix(single_df, valid_groups)
    matrix.to_csv(os.path.join(output_dir, f'robust_rq3_cross_{group_label}_matrix_{sort_by}_{SCHEME}.csv'))
    print(f"  AUC 矩阵 (行=train, 列=test):\n{matrix.round(3).to_string()}")

    combined_df = run_combined_to_single(data, group_col, valid_groups, features, best_params)
    combined_df.to_csv(os.path.join(output_dir, f'robust_rq3_cross_{group_label}_combined_{sort_by}_{SCHEME}.csv'), index=False)

    return {'within': within_df, 'single': single_df, 'matrix': matrix, 'combined': combined_df,
            'valid_groups': valid_groups}


# ============================================================
# Main
# ============================================================

def main():
    output_dir = script_dir

    for sort_by in ['downloads', 'likes']:
        print(f"\n{'='*60}")
        print(f"Robustness Check: by {sort_by}")
        print(f"{'='*60}")

        data_path = os.path.join(script_dir, f'robust_filtered_by_{sort_by}_{SCHEME}.csv')
        data = pd.read_csv(data_path)
        data = data[data['category'].isin(['popular', 'unpopular'])]
        print(f"数据量: {len(data)} 条 (popular={int((data['category']=='popular').sum())}, "
              f"unpopular={int((data['category']=='unpopular').sum())})")

        run_rq1(data, sort_by, output_dir)
        _, rf_best_params, selected_features, rf_best_estimator, best_model_name, best_auc = \
            run_rq2(data, sort_by, output_dir)

        # Permutation importance (在 best model 上, 跟 baseline 对齐)
        run_permutation_importance(data, sort_by, rf_best_estimator, best_model_name, best_auc,
                                   selected_features, output_dir)

        if rf_best_params is None:
            print(f"  ⚠️ 没拿到 RF best_params,RQ3 跳过")
            continue
        print(f"\n  RQ3 使用的 RF best_params (来自 robust RQ2): {rf_best_params}")

        run_rq3_cross_group(data, sort_by, 'domain', 'domain',
                            selected_features, rf_best_params, output_dir)
        run_rq3_cross_group(data, sort_by, 'affiliation', 'affiliation',
                            selected_features, rf_best_params, output_dir)

    print(f"\n{'='*60}")
    print("✅ Robustness check 全部完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
