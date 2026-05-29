import pandas as pd
import numpy as np
import os
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
RF_FEATURES_DIR = os.path.join(script_dir, '../random-forest')

# 所有 cross-group 阶段产出的 csv/md 都放这个子文件夹
CROSS_OUTPUT_DIR = os.path.join(script_dir, 'cross_group_outputs')
os.makedirs(CROSS_OUTPUT_DIR, exist_ok=True)

MIN_SAMPLES = 30          # 一个 group 中 popular 和 unpopular 都需要至少这么多
BOOTSTRAP_N = 1000        # bootstrap CI 迭代数
RNG_SEED = 42


def load_rq2_best_params(sort_by):
    """读 RQ2 跑出来的 best RF 超参 (跟论文承诺 'reuse best classifier from RQ2' 对齐)"""
    info = pd.read_csv(os.path.join(RF_FEATURES_DIR, f'best_model_{sort_by}_{SCHEME}.csv')).iloc[0]
    params = ast.literal_eval(info['best_params'])
    return {k.replace('clf__', ''): v for k, v in params.items()}


def load_selected_features(sort_by):
    path = os.path.join(RF_FEATURES_DIR, f'selected_features_{sort_by}_{SCHEME}.csv')
    return pd.read_csv(path)['feature'].tolist()


def make_rf(best_params):
    # class_weight='balanced' 是为了应对 popular:unpopular ~ 1:8 的类不平衡
    # 跟 baseline (../random-forest/ml_classifiers.py) 保持一致
    return RandomForestClassifier(random_state=RNG_SEED, n_jobs=-1, class_weight='balanced', **best_params)


def bootstrap_auc_ci(y_true, y_pred_prob, n_iter=BOOTSTRAP_N, rng_seed=RNG_SEED):
    """对 (y_true, y_pred_prob) bootstrap 重采样,返回 AUC mean / std / 2.5% / 97.5%"""
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue   # 该次重采样只有一个类,AUC 算不出来,跳过
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
            print(f"  ⚠️ 跳过 {group} (popular={pop}, unpopular={unpop}, 不足 {MIN_SAMPLES})")
    return sorted(valid)


# -------- within-group baseline (10-fold CV in same group) --------

def run_within_group_baseline(data, group_col, valid_groups, features, best_params):
    rows = []
    print("\n--- Within-group baseline (10-fold CV in same group) ---")
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
        print(f"  {group}: AUC={aucs.mean():.4f} ± {aucs.std():.4f} (n={len(g)})")
        rows.append({
            'Group': group, 'N': len(g),
            'AUC_mean': aucs.mean(), 'AUC_std': aucs.std(),
        })
    return pd.DataFrame(rows)


# -------- 1-vs-1 cross-group (with bootstrap CI on test AUC) --------

def run_single_to_single(data, group_col, valid_groups, features, best_params):
    rows = []
    print("\n--- 单对单 cross-group (train on A → test on B), bootstrap 1000 CI ---")
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
            print(f"  {tr} → {te}: AUC={mean_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
                  f"(N_train={len(tr_data)}, N_test={len(te_data)})")
            rows.append({
                'Train': tr, 'Test': te,
                'AUC_mean': mean_auc, 'AUC_std': std_auc,
                'CI_lo': ci_lo, 'CI_hi': ci_hi,
                'N_train': len(tr_data), 'N_test': len(te_data),
            })
    return pd.DataFrame(rows)


# -------- others-vs-1 cross-group --------

def run_combined_to_single(data, group_col, valid_groups, features, best_params):
    rows = []
    print("\n--- 多对一 cross-group (train on all-others → test on target) ---")
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
        print(f"  others → {te}: AUC={mean_auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"(N_train={len(tr_data)}, N_test={len(te_data)})")
        rows.append({
            'Train': 'others',
            'Test': te,
            'AUC_mean': mean_auc, 'AUC_std': std_auc,
            'CI_lo': ci_lo, 'CI_hi': ci_hi,
            'N_train': len(tr_data), 'N_test': len(te_data),
        })
    return pd.DataFrame(rows)


def make_auc_matrix(single_df, valid_groups):
    m = pd.DataFrame(index=valid_groups, columns=valid_groups, dtype=float)
    for _, r in single_df.iterrows():
        m.loc[r['Train'], r['Test']] = r['AUC_mean']
    for g in valid_groups:
        m.loc[g, g] = np.nan
    return m


# -------- 一个 (sort_by, group_label) 的完整流程 --------

def run_cross_analysis(sort_by, group_col, group_label):
    print(f"\n{'='*60}")
    print(f"Cross-{group_label}: by {sort_by}")
    print(f"{'='*60}")

    data_path = os.path.join(script_dir, 'mapping_outputs', f'data_with_domain_affiliation_{sort_by}_{SCHEME}.csv')
    data = pd.read_csv(data_path)
    data = data[data['category'].isin(['popular', 'unpopular'])]
    if group_col not in data.columns:
        print(f"⚠️ 列 {group_col} 不存在,跳过")
        return None

    features = load_selected_features(sort_by)
    best_params = load_rq2_best_params(sort_by)
    print(f"使用特征数: {len(features)}, RF 参数: {best_params}, 数据量: {len(data)}")

    print(f"\n有效 {group_label} groups (每类 ≥ {MIN_SAMPLES} 样本):")
    valid_groups = get_valid_groups(data, group_col)
    print(f"共 {len(valid_groups)} 个: {valid_groups}")

    if len(valid_groups) < 2:
        print("⚠️ 有效 group 不足 2 个,跳过")
        return None

    within_df = run_within_group_baseline(data, group_col, valid_groups, features, best_params)
    within_df.to_csv(os.path.join(CROSS_OUTPUT_DIR, f'within_{group_label}_{sort_by}_{SCHEME}.csv'), index=False)

    single_df = run_single_to_single(data, group_col, valid_groups, features, best_params)
    single_df.to_csv(os.path.join(CROSS_OUTPUT_DIR, f'cross_{group_label}_single_{sort_by}_{SCHEME}.csv'), index=False)

    matrix = make_auc_matrix(single_df, valid_groups)
    matrix.to_csv(os.path.join(CROSS_OUTPUT_DIR, f'cross_{group_label}_matrix_{sort_by}_{SCHEME}.csv'))
    print(f"\nAUC 矩阵 (行=训练 group, 列=测试 group):")
    print(matrix.round(3).to_string())

    combined_df = run_combined_to_single(data, group_col, valid_groups, features, best_params)
    combined_df.to_csv(os.path.join(CROSS_OUTPUT_DIR, f'cross_{group_label}_combined_{sort_by}_{SCHEME}.csv'), index=False)

    return {
        'sort_by': sort_by, 'group_label': group_label,
        'valid_groups': valid_groups,
        'within': within_df, 'single': single_df, 'matrix': matrix,
        'combined': combined_df,
    }


# -------- md 摘要 --------

def fmt_auc_ci(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def generate_md_summary(all_results):
    md = []
    md.append(f"# RQ3: Cross-group Analysis Summary (scheme {SCHEME})")
    md.append("")
    md.append("## 方法学")
    md.append("")
    md.append("- **Best classifier from RQ2**: RandomForest, 用 `best_model_<sort_by>_10-10-80.csv` 里存的最优超参")
    md.append("- **特征**: RQ2 step 2 选出的特征 (`selected_features_<sort_by>_10-10-80.csv`, 26 个)")
    md.append("- **分组**:")
    md.append("  - Domain: `pipeline_content` 映射到 6 类 (Audio / Computer Vision / Multimodal / NLP / Tabular / Reinforcement Learning); 未映射的归 'Other' (实验中视情况保留或排除)")
    md.append("  - Affiliation: 解析 HF profile page HTML 拿 `span.capitalize`; 默认 fallback `organization or individual`")
    md.append(f"- **最小样本阈值**: 一个 group 中 popular 和 unpopular 各 ≥ {MIN_SAMPLES}")
    md.append("- **Within-group baseline**: 10-fold CV 在同组内 (AUC ± std)")
    md.append("- **Cross-group 1-vs-1**: train on group A, test on group B (A ≠ B)")
    md.append("- **Cross-group others-vs-1**: train on union of all-other-groups, test on target group")
    md.append(f"- **AUC 不确定性**: 在 test set 上 bootstrap 重采样 {BOOTSTRAP_N} 次, 报 mean + 95% CI (2.5%–97.5% percentile)")
    md.append("")

    for r in all_results:
        if r is None:
            continue
        sort_by = r['sort_by']
        gl = r['group_label']

        md.append(f"## by {sort_by} — Cross-{gl}")
        md.append("")
        md.append(f"### Within-group baseline ({len(r['valid_groups'])} groups)")
        md.append("")
        md.append("| Group | N | AUC (10-fold CV) |")
        md.append("| --- | --- | --- |")
        for _, row in r['within'].iterrows():
            md.append(f"| {row['Group']} | {row['N']} | {row['AUC_mean']:.3f} ± {row['AUC_std']:.3f} |")
        md.append("")

        md.append(f"### Cross-group 1-vs-1 AUC matrix (行=train, 列=test)")
        md.append("")
        md.append("| Train \\ Test | " + " | ".join(r['valid_groups']) + " |")
        md.append("| --- |" + " --- |" * len(r['valid_groups']))
        for tr in r['valid_groups']:
            cells = []
            for te in r['valid_groups']:
                v = r['matrix'].loc[tr, te]
                cells.append("—" if pd.isna(v) else f"{v:.3f}")
            md.append(f"| **{tr}** | " + " | ".join(cells) + " |")
        md.append("")
        md.append("注: 数值是 bootstrap AUC mean, CI 见 `cross_<label>_single_<sort_by>_10-10-80.csv`")
        md.append("")

        md.append(f"### Others-vs-1 (general → specific group)")
        md.append("")
        md.append("| Test group | N_test | AUC mean | 95% CI |")
        md.append("| --- | --- | --- | --- |")
        for _, row in r['combined'].iterrows():
            md.append(f"| {row['Test']} | {row['N_test']} | {row['AUC_mean']:.3f} | "
                      f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] |")
        md.append("")

        # 关键观察: 与 within-group baseline 对比, 看跨组掉了多少
        within_map = {row['Group']: row['AUC_mean'] for _, row in r['within'].iterrows()}
        md.append("### 跨组 vs 同组 baseline 对比 (评估泛化能力)")
        md.append("")
        md.append("对每个 test group, 列同组 baseline AUC vs 跨组平均 AUC (1-vs-1 平均):")
        md.append("")
        md.append("| Test group | Within (baseline) | Cross 1-vs-1 mean | Δ (cross − within) |")
        md.append("| --- | --- | --- | --- |")
        for g in r['valid_groups']:
            baseline = within_map.get(g, np.nan)
            cross_mean = r['single'][r['single']['Test'] == g]['AUC_mean'].mean()
            delta = cross_mean - baseline
            md.append(f"| {g} | {baseline:.3f} | {cross_mean:.3f} | {delta:+.3f} |")
        md.append("")

    summary_path = os.path.join(script_dir, f'rq3_summary_{SCHEME}.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f"\n📝 汇总已写入: {summary_path}")


if __name__ == '__main__':
    all_results = []
    for sort_by in ['downloads', 'likes']:
        all_results.append(run_cross_analysis(sort_by, 'domain', 'domain'))
        all_results.append(run_cross_analysis(sort_by, 'affiliation', 'affiliation'))

    generate_md_summary(all_results)
    print(f"\n{'='*60}\n✅ 所有 Cross-group 分析完成!\n{'='*60}")
