"""
对比 robust check (去掉 downloads top 1%) 与主分析的结果。

产出:
  - rq1_diff_<sort_by>_10-10-80.csv
  - rq2_diff_10-10-80.csv
  - rq3_within_diff_<group>_<sort_by>_10-10-80.csv
  - rq3_single_diff_<group>_<sort_by>_10-10-80.csv
  - rq3_combined_diff_<group>_<sort_by>_10-10-80.csv
  - feature_importance_diff_perm_<sort_by>_10-10-80.csv
  - feature_importance_diff_impurity_<sort_by>_10-10-80.csv
  - comparison_with_main_10-10-80.md   (汇总, 可直接贴进 README)
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'

MAIN_RQ1_DIR = os.path.join(script_dir, '../../research-qs-analysis/significant-analyze')
MAIN_RQ2_DIR = os.path.join(script_dir, '../../research-qs-analysis/random-forest')
MAIN_RQ3_DIR = os.path.join(script_dir, '../../research-qs-analysis/cross-grouping/cross_group_outputs')
ROBUST_DIR = script_dir
OUT_DIFF_DIR = os.path.join(script_dir, 'comparison_outputs')
os.makedirs(OUT_DIFF_DIR, exist_ok=True)

# 判定阈值
RQ2_AUC_DELTA_FLAG = 0.02     # |ΔAUC| > 0.02 标 "noticeable"
RQ3_AUC_DELTA_FLAG = 0.03
TOP_K_OVERLAP = 10            # 比较两边 top-10 特征重合度


# ============================================================
# RQ1
# ============================================================

def diff_rq1(sort_by):
    main = pd.read_csv(os.path.join(MAIN_RQ1_DIR, f'significance_analysis_by_{sort_by}_{SCHEME}.csv'))
    rob = pd.read_csv(os.path.join(ROBUST_DIR, f'robust_rq1_significance_{sort_by}_{SCHEME}.csv'))

    keep = ['Feature', 'p_value_fdr_corrected', 'Effect_Size_Value', 'Effect_Size_Interpretation', 'significant_after_fdr']
    m = main[keep].add_suffix('_main').rename(columns={'Feature_main': 'Feature'})
    r = rob[keep].add_suffix('_robust').rename(columns={'Feature_robust': 'Feature'})
    df = m.merge(r, on='Feature', how='inner')

    df['delta_effect'] = df['Effect_Size_Value_robust'] - df['Effect_Size_Value_main']
    df['sig_flip'] = df['significant_after_fdr_main'] != df['significant_after_fdr_robust']
    df['category_change'] = df['Effect_Size_Interpretation_main'] != df['Effect_Size_Interpretation_robust']

    out_path = os.path.join(OUT_DIFF_DIR, f'rq1_diff_{sort_by}_{SCHEME}.csv')
    df.to_csv(out_path, index=False)

    n_total = len(df)
    n_sig_main = int(df['significant_after_fdr_main'].sum())
    n_sig_rob = int(df['significant_after_fdr_robust'].sum())
    n_flip = int(df['sig_flip'].sum())
    n_cat_change = int(df['category_change'].sum())
    max_abs_delta = df['delta_effect'].abs().max()

    return {
        'sort_by': sort_by,
        'n_features': n_total,
        'n_sig_main': n_sig_main,
        'n_sig_robust': n_sig_rob,
        'n_sig_flip': n_flip,
        'n_category_change': n_cat_change,
        'max_abs_delta_effect': max_abs_delta,
        'flipped_features': df[df['sig_flip']]['Feature'].tolist(),
        'category_changed_features': df[df['category_change']][
            ['Feature', 'Effect_Size_Interpretation_main', 'Effect_Size_Interpretation_robust']
        ].to_dict('records'),
    }


# ============================================================
# RQ2
# ============================================================

def diff_rq2():
    rows = []
    for sb in ['downloads', 'likes']:
        main = pd.read_csv(os.path.join(MAIN_RQ2_DIR, f'ml_classifier_results_{sb}_{SCHEME}.csv'))
        rob = pd.read_csv(os.path.join(ROBUST_DIR, f'robust_rq2_classifiers_{sb}_{SCHEME}.csv'))
        for _, mrow in main.iterrows():
            rrow = rob[rob['Model'] == mrow['Model']]
            if rrow.empty:
                continue
            rrow = rrow.iloc[0]
            delta = rrow['AUC_mean'] - mrow['AUC_mean']
            rows.append({
                'sort_by': sb,
                'Model': mrow['Model'],
                'AUC_main': mrow['AUC_mean'],
                'AUC_robust': rrow['AUC_mean'],
                'delta_AUC': delta,
                'best_params_main': mrow['Best_Params'],
                'best_params_robust': rrow['Best_Params'],
                'params_changed': mrow['Best_Params'] != rrow['Best_Params'],
                'noticeable_drop': abs(delta) > RQ2_AUC_DELTA_FLAG,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIFF_DIR, f'rq2_diff_{SCHEME}.csv'), index=False)
    return df


# ============================================================
# RQ3
# ============================================================

def diff_rq3_within(sort_by, group_label):
    main_path = os.path.join(MAIN_RQ3_DIR, f'within_{group_label}_{sort_by}_{SCHEME}.csv')
    rob_path = os.path.join(ROBUST_DIR, f'robust_rq3_within_{group_label}_{sort_by}_{SCHEME}.csv')
    if not (os.path.exists(main_path) and os.path.exists(rob_path)):
        return None
    main = pd.read_csv(main_path)
    rob = pd.read_csv(rob_path)
    df = main.merge(rob, on='Group', suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIFF_DIR, f'rq3_within_diff_{group_label}_{sort_by}_{SCHEME}.csv'), index=False)
    return df


def diff_rq3_single(sort_by, group_label):
    """1-vs-1 cross-group: 比较 ΔAUC + 检查 main 是否落在 robust CI 内"""
    main_path = os.path.join(MAIN_RQ3_DIR, f'cross_{group_label}_single_{sort_by}_{SCHEME}.csv')
    rob_path = os.path.join(ROBUST_DIR, f'robust_rq3_cross_{group_label}_single_{sort_by}_{SCHEME}.csv')
    if not (os.path.exists(main_path) and os.path.exists(rob_path)):
        return None
    main = pd.read_csv(main_path)
    rob = pd.read_csv(rob_path)
    df = main.merge(rob, on=['Train', 'Test'], suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['main_in_robust_CI'] = (df['AUC_mean_main'] >= df['CI_lo_robust']) & \
                              (df['AUC_mean_main'] <= df['CI_hi_robust'])
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIFF_DIR, f'rq3_single_diff_{group_label}_{sort_by}_{SCHEME}.csv'), index=False)
    return df


def diff_rq3_combined(sort_by, group_label):
    main_path = os.path.join(MAIN_RQ3_DIR, f'cross_{group_label}_combined_{sort_by}_{SCHEME}.csv')
    rob_path = os.path.join(ROBUST_DIR, f'robust_rq3_cross_{group_label}_combined_{sort_by}_{SCHEME}.csv')
    if not (os.path.exists(main_path) and os.path.exists(rob_path)):
        return None
    main = pd.read_csv(main_path)
    rob = pd.read_csv(rob_path)
    df = main.merge(rob, on='Test', suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['main_in_robust_CI'] = (df['AUC_mean_main'] >= df['CI_lo_robust']) & \
                              (df['AUC_mean_main'] <= df['CI_hi_robust'])
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIFF_DIR, f'rq3_combined_diff_{group_label}_{sort_by}_{SCHEME}.csv'), index=False)
    return df


# ============================================================
# Feature Importance Ranking — verify ranking stability
# ============================================================

def diff_feature_importance(sort_by, kind):
    """对比 main vs robust 的特征排名稳定性。
    kind: 'perm' (permutation_importance) 或 'impurity' (RF impurity-based)

    输出: 每特征 (main 排名, robust 排名, 排名差, 重要性 main, 重要性 robust, ΔImportance)
    summary 指标: Spearman rank correlation, top-K overlap
    """
    if kind == 'perm':
        main_path = os.path.join(MAIN_RQ2_DIR, f'permutation_importance_{sort_by}_{SCHEME}.csv')
        rob_path = os.path.join(ROBUST_DIR, f'robust_permutation_importance_{sort_by}_{SCHEME}.csv')
        value_col = 'PermImportance_mean'
        out_name = f'feature_importance_diff_perm_{sort_by}_{SCHEME}.csv'
    elif kind == 'impurity':
        main_path = os.path.join(MAIN_RQ2_DIR, f'rf_impurity_importance_{sort_by}_{SCHEME}.csv')
        rob_path = os.path.join(ROBUST_DIR, f'robust_rf_impurity_importance_{sort_by}_{SCHEME}.csv')
        value_col = 'Importance'
        out_name = f'feature_importance_diff_impurity_{sort_by}_{SCHEME}.csv'
    else:
        raise ValueError(f"unknown kind: {kind}")

    if not (os.path.exists(main_path) and os.path.exists(rob_path)):
        print(f"  ⚠️ {kind} 文件缺失 (sort_by={sort_by}), 跳过")
        return None

    main = pd.read_csv(main_path).reset_index(drop=True)
    rob = pd.read_csv(rob_path).reset_index(drop=True)

    # 两边按 importance 降序排好, 取 rank (1 = 最重要)
    main = main.sort_values(value_col, ascending=False).reset_index(drop=True)
    rob = rob.sort_values(value_col, ascending=False).reset_index(drop=True)
    main['rank_main'] = main.index + 1
    rob['rank_robust'] = rob.index + 1
    main = main[['Feature', value_col, 'rank_main']].rename(columns={value_col: f'{value_col}_main'})
    rob = rob[['Feature', value_col, 'rank_robust']].rename(columns={value_col: f'{value_col}_robust'})

    df = main.merge(rob, on='Feature', how='inner').sort_values('rank_main').reset_index(drop=True)
    df['rank_delta'] = df['rank_robust'] - df['rank_main']
    df['importance_delta'] = df[f'{value_col}_robust'] - df[f'{value_col}_main']

    df.to_csv(os.path.join(OUT_DIFF_DIR, out_name), index=False)

    # Spearman rank correlation
    rho, p = spearmanr(df['rank_main'], df['rank_robust'])

    # Top-K overlap
    top_main = set(df.nsmallest(TOP_K_OVERLAP, 'rank_main')['Feature'])
    top_rob = set(df.nsmallest(TOP_K_OVERLAP, 'rank_robust')['Feature'])
    overlap = len(top_main & top_rob)

    max_abs_rank_delta = int(df['rank_delta'].abs().max())
    mean_abs_rank_delta = float(df['rank_delta'].abs().mean())

    return {
        'sort_by': sort_by,
        'kind': kind,
        'n_features': len(df),
        'spearman_rho': float(rho),
        'spearman_p': float(p),
        'topK': TOP_K_OVERLAP,
        'topK_overlap': overlap,
        'max_abs_rank_delta': max_abs_rank_delta,
        'mean_abs_rank_delta': mean_abs_rank_delta,
    }


# ============================================================
# Markdown report
# ============================================================

def write_md(rq1_results, rq2_df, rq3_data, feat_imp_results):
    md = []
    md.append(f"# Robustness Check vs Main Analysis — Comparison (scheme {SCHEME})")
    md.append("")
    md.append("> 对比对象: 主分析 vs 去掉 `downloads` top 1% 后重跑的 robustness analysis。")
    md.append(f"> 阈值: RQ2 |ΔAUC| > {RQ2_AUC_DELTA_FLAG}, RQ3 |ΔAUC| > {RQ3_AUC_DELTA_FLAG} 视为 noticeable。")
    md.append("")

    # ---------- RQ1 ----------
    md.append("## RQ1 — Significance Analysis")
    md.append("")
    md.append("| sort_by | features | sig (main) | sig (robust) | significance flips | effect-size category changes | max \\|Δeffect\\| |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rq1_results:
        md.append(f"| {r['sort_by']} | {r['n_features']} | {r['n_sig_main']} | {r['n_sig_robust']} | "
                  f"{r['n_sig_flip']} | {r['n_category_change']} | {r['max_abs_delta_effect']:.4f} |")
    md.append("")
    for r in rq1_results:
        if r['n_sig_flip'] > 0 or r['n_category_change'] > 0:
            md.append(f"### by {r['sort_by']} — 变化细节")
            md.append("")
            if r['n_sig_flip'] > 0:
                md.append(f"**Significance flipped ({r['n_sig_flip']})**: " + ", ".join(r['flipped_features']))
                md.append("")
            if r['n_category_change'] > 0:
                md.append("**Effect-size category changed:**")
                md.append("")
                md.append("| Feature | main | robust |")
                md.append("| --- | --- | --- |")
                for c in r['category_changed_features']:
                    md.append(f"| {c['Feature']} | {c['Effect_Size_Interpretation_main']} | "
                              f"{c['Effect_Size_Interpretation_robust']} |")
                md.append("")

    # ---------- RQ2 ----------
    md.append("## RQ2 — ML Classifiers")
    md.append("")
    md.append("| sort_by | Model | AUC (main) | AUC (robust) | ΔAUC | params changed? | noticeable? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in rq2_df.iterrows():
        flag = "**yes**" if row['noticeable_drop'] else "no"
        pc = "yes" if row['params_changed'] else "no"
        md.append(f"| {row['sort_by']} | {row['Model']} | {row['AUC_main']:.4f} | {row['AUC_robust']:.4f} | "
                  f"{row['delta_AUC']:+.4f} | {pc} | {flag} |")
    md.append("")
    n_notice = int(rq2_df['noticeable_drop'].sum())
    n_params = int(rq2_df['params_changed'].sum())
    md.append(f"**Summary**: {n_notice}/{len(rq2_df)} (model × sort_by) pairs 的 ΔAUC 超过 {RQ2_AUC_DELTA_FLAG}; "
              f"{n_params}/{len(rq2_df)} 的 best params 发生变化。")
    md.append("")

    # ---------- RQ3 ----------
    md.append("## RQ3 — Cross-group Analysis")
    md.append("")

    # Within-group
    md.append("### Within-group baseline (10-fold CV)")
    md.append("")
    md.append("| sort_by | group_label | group | AUC main | AUC robust | ΔAUC | noticeable? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), w in rq3_data['within'].items():
        if w is None: continue
        for _, row in w.iterrows():
            flag = "**yes**" if row['noticeable'] else "no"
            md.append(f"| {sb} | {gl} | {row['Group']} | {row['AUC_mean_main']:.4f} | "
                      f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | {flag} |")
    md.append("")

    # 1-vs-1: 给个 summary + flag 的 cell
    md.append("### Cross-group 1-vs-1 (train A → test B)")
    md.append("")
    md.append("| sort_by | group_label | N cells | mean ΔAUC | max \\|ΔAUC\\| | cells with main outside robust CI | noticeable cells |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), s in rq3_data['single'].items():
        if s is None: continue
        n = len(s)
        mean_delta = s['delta_AUC'].mean()
        max_abs = s['delta_AUC'].abs().max()
        out_ci = int((~s['main_in_robust_CI']).sum())
        notice = int(s['noticeable'].sum())
        md.append(f"| {sb} | {gl} | {n} | {mean_delta:+.4f} | {max_abs:.4f} | {out_ci} | {notice} |")
    md.append("")
    # 列出 noticeable 的 cells
    for (sb, gl), s in rq3_data['single'].items():
        if s is None: continue
        bad = s[s['noticeable']]
        if not bad.empty:
            md.append(f"#### by {sb} / cross-{gl} — noticeable cells ({len(bad)})")
            md.append("")
            md.append("| Train → Test | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for _, row in bad.iterrows():
                in_ci = "yes" if row['main_in_robust_CI'] else "**no**"
                md.append(f"| {row['Train']} → {row['Test']} | {row['AUC_mean_main']:.4f} | "
                          f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | "
                          f"[{row['CI_lo_robust']:.3f}, {row['CI_hi_robust']:.3f}] | {in_ci} |")
            md.append("")

    # ---------- Feature Importance Ranking Stability ----------
    if feat_imp_results:
        md.append("## Feature Importance — Ranking Stability")
        md.append("")
        md.append("> 比较主分析 vs robust 上同一 RF best model 的特征重要性排名。")
        md.append(f"> 指标: Spearman 排名相关系数 (1.0 = 完全一致), top-{TOP_K_OVERLAP} 特征重合度, 排名 delta。")
        md.append("")
        md.append("| sort_by | importance kind | n_features | Spearman ρ | p | top-K overlap | max \\|Δrank\\| | mean \\|Δrank\\| |")
        md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in feat_imp_results:
            if r is None:
                continue
            md.append(f"| {r['sort_by']} | {r['kind']} | {r['n_features']} | "
                      f"{r['spearman_rho']:+.4f} | {r['spearman_p']:.2e} | "
                      f"{r['topK_overlap']}/{r['topK']} | {r['max_abs_rank_delta']} | "
                      f"{r['mean_abs_rank_delta']:.2f} |")
        md.append("")
        md.append("**解读**: Spearman ρ 越接近 1.0 + top-K overlap 越接近 n / max |Δrank| 越小, 说明特征排名越稳健。")
        md.append("")

    # Others-vs-1
    md.append("### Cross-group others-vs-1 (others → target)")
    md.append("")
    md.append("| sort_by | group_label | target | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), c in rq3_data['combined'].items():
        if c is None: continue
        for _, row in c.iterrows():
            in_ci = "yes" if row['main_in_robust_CI'] else "**no**"
            md.append(f"| {sb} | {gl} | {row['Test']} | {row['AUC_mean_main']:.4f} | "
                      f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | "
                      f"[{row['CI_lo_robust']:.3f}, {row['CI_hi_robust']:.3f}] | {in_ci} |")
    md.append("")

    # 写出
    out_md = os.path.join(script_dir, f'comparison_with_main_{SCHEME}.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f"对比汇总: {out_md}")


def main():
    print("=== RQ1 ===")
    rq1_results = [diff_rq1('downloads'), diff_rq1('likes')]
    for r in rq1_results:
        print(f"  by {r['sort_by']}: sig {r['n_sig_main']} → {r['n_sig_robust']}, "
              f"flips={r['n_sig_flip']}, category_change={r['n_category_change']}, "
              f"max |Δeffect|={r['max_abs_delta_effect']:.4f}")

    print("\n=== RQ2 ===")
    rq2_df = diff_rq2()
    print(rq2_df[['sort_by', 'Model', 'AUC_main', 'AUC_robust', 'delta_AUC',
                  'params_changed', 'noticeable_drop']].to_string(index=False))

    print("\n=== RQ3 ===")
    rq3_data = {'within': {}, 'single': {}, 'combined': {}}
    for sb in ['downloads', 'likes']:
        for gl in ['domain', 'affiliation']:
            rq3_data['within'][(sb, gl)] = diff_rq3_within(sb, gl)
            rq3_data['single'][(sb, gl)] = diff_rq3_single(sb, gl)
            rq3_data['combined'][(sb, gl)] = diff_rq3_combined(sb, gl)
            s = rq3_data['single'][(sb, gl)]
            c = rq3_data['combined'][(sb, gl)]
            if s is not None:
                n_out_ci = int((~s['main_in_robust_CI']).sum())
                print(f"  {sb}/{gl} 1-vs-1: {len(s)} cells, mean ΔAUC={s['delta_AUC'].mean():+.4f}, "
                      f"max |Δ|={s['delta_AUC'].abs().max():.4f}, "
                      f"main outside CI: {n_out_ci}, noticeable: {int(s['noticeable'].sum())}")
            if c is not None:
                n_out_ci = int((~c['main_in_robust_CI']).sum())
                print(f"  {sb}/{gl} others-vs-1: {len(c)} cells, mean ΔAUC={c['delta_AUC'].mean():+.4f}, "
                      f"main outside CI: {n_out_ci}")

    print("\n=== Feature Importance Ranking Stability ===")
    feat_imp_results = []
    for sb in ['downloads', 'likes']:
        for kind in ['perm', 'impurity']:
            r = diff_feature_importance(sb, kind)
            if r is not None:
                print(f"  {sb}/{kind}: Spearman ρ={r['spearman_rho']:+.4f}, "
                      f"top-{r['topK']} overlap={r['topK_overlap']}/{r['topK']}, "
                      f"max |Δrank|={r['max_abs_rank_delta']}, mean |Δrank|={r['mean_abs_rank_delta']:.2f}")
            feat_imp_results.append(r)

    write_md(rq1_results, rq2_df, rq3_data, feat_imp_results)


if __name__ == '__main__':
    main()
