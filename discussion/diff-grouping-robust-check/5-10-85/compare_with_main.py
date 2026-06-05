"""
对比当前 scheme (5-10-85) 与主分析 (10-10-80) 的 RQ1 / RQ2 / RQ3 结果。

产出:
  - comparison_outputs/rq1_diff_<sort_by>.csv
  - comparison_outputs/rq2_diff.csv
  - comparison_outputs/rq3_*_diff_<group>_<sort_by>.csv
  - comparison_with_main.md  (汇总, 可直接贴进 README)
"""
import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '5-10-85'
MAIN_SCHEME = '10-10-80'

MAIN_RQ1 = os.path.join(script_dir, '../../../research-qs-analysis/significant-analyze')
MAIN_RQ2 = os.path.join(script_dir, '../../../research-qs-analysis/random-forest')
MAIN_RQ3 = os.path.join(script_dir, '../../../research-qs-analysis/cross-grouping/cross_group_outputs')
OUT_DIR = os.path.join(script_dir, 'comparison_outputs')
os.makedirs(OUT_DIR, exist_ok=True)

RQ2_AUC_DELTA_FLAG = 0.02
RQ3_AUC_DELTA_FLAG = 0.03


def diff_rq1(sort_by):
    main = pd.read_csv(os.path.join(MAIN_RQ1, f'significance_analysis_by_{sort_by}_{MAIN_SCHEME}.csv'))
    rob = pd.read_csv(os.path.join(script_dir, f'rq1_significance_{sort_by}_{SCHEME}.csv'))

    keep = ['Feature', 'p_value_fdr_corrected', 'Effect_Size_Value',
            'Effect_Size_Interpretation', 'significant_after_fdr']
    m = main[keep].add_suffix('_main').rename(columns={'Feature_main': 'Feature'})
    r = rob[keep].add_suffix('_robust').rename(columns={'Feature_robust': 'Feature'})
    df = m.merge(r, on='Feature', how='inner')

    df['delta_effect'] = df['Effect_Size_Value_robust'] - df['Effect_Size_Value_main']
    df['sig_flip'] = df['significant_after_fdr_main'] != df['significant_after_fdr_robust']
    df['category_change'] = df['Effect_Size_Interpretation_main'] != df['Effect_Size_Interpretation_robust']
    df.to_csv(os.path.join(OUT_DIR, f'rq1_diff_{sort_by}.csv'), index=False)

    return {
        'sort_by': sort_by,
        'n_features': len(df),
        'n_sig_main': int(df['significant_after_fdr_main'].sum()),
        'n_sig_robust': int(df['significant_after_fdr_robust'].sum()),
        'n_sig_flip': int(df['sig_flip'].sum()),
        'n_category_change': int(df['category_change'].sum()),
        'max_abs_delta_effect': df['delta_effect'].abs().max(),
        'flipped_features': df[df['sig_flip']]['Feature'].tolist(),
        'category_changed_features': df[df['category_change']][
            ['Feature', 'Effect_Size_Interpretation_main', 'Effect_Size_Interpretation_robust']
        ].to_dict('records'),
    }


def diff_rq2():
    rows = []
    for sb in ['downloads', 'likes']:
        main = pd.read_csv(os.path.join(MAIN_RQ2, f'ml_classifier_results_{sb}_{MAIN_SCHEME}.csv'))
        rob = pd.read_csv(os.path.join(script_dir, f'rq2_classifiers_{sb}_{SCHEME}.csv'))
        for _, mrow in main.iterrows():
            rrow = rob[rob['Model'] == mrow['Model']]
            if rrow.empty: continue
            rrow = rrow.iloc[0]
            delta = rrow['AUC_mean'] - mrow['AUC_mean']
            rows.append({
                'sort_by': sb, 'Model': mrow['Model'],
                'AUC_main': mrow['AUC_mean'], 'AUC_robust': rrow['AUC_mean'],
                'delta_AUC': delta,
                'best_params_main': mrow['Best_Params'],
                'best_params_robust': rrow['Best_Params'],
                'params_changed': mrow['Best_Params'] != rrow['Best_Params'],
                'noticeable_drop': abs(delta) > RQ2_AUC_DELTA_FLAG,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'rq2_diff.csv'), index=False)
    return df


def diff_rq3_within(sb, gl):
    main_p = os.path.join(MAIN_RQ3, f'within_{gl}_{sb}_{MAIN_SCHEME}.csv')
    rob_p = os.path.join(script_dir, f'rq3_within_{gl}_{sb}_{SCHEME}.csv')
    if not (os.path.exists(main_p) and os.path.exists(rob_p)): return None
    main = pd.read_csv(main_p)
    rob = pd.read_csv(rob_p)
    df = main.merge(rob, on='Group', suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIR, f'rq3_within_diff_{gl}_{sb}.csv'), index=False)
    return df


def diff_rq3_single(sb, gl):
    main_p = os.path.join(MAIN_RQ3, f'cross_{gl}_single_{sb}_{MAIN_SCHEME}.csv')
    rob_p = os.path.join(script_dir, f'rq3_cross_{gl}_single_{sb}_{SCHEME}.csv')
    if not (os.path.exists(main_p) and os.path.exists(rob_p)): return None
    main = pd.read_csv(main_p)
    rob = pd.read_csv(rob_p)
    df = main.merge(rob, on=['Train', 'Test'], suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['main_in_robust_CI'] = (df['AUC_mean_main'] >= df['CI_lo_robust']) & \
                              (df['AUC_mean_main'] <= df['CI_hi_robust'])
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIR, f'rq3_single_diff_{gl}_{sb}.csv'), index=False)
    return df


def diff_rq3_combined(sb, gl):
    main_p = os.path.join(MAIN_RQ3, f'cross_{gl}_combined_{sb}_{MAIN_SCHEME}.csv')
    rob_p = os.path.join(script_dir, f'rq3_cross_{gl}_combined_{sb}_{SCHEME}.csv')
    if not (os.path.exists(main_p) and os.path.exists(rob_p)): return None
    main = pd.read_csv(main_p)
    rob = pd.read_csv(rob_p)
    df = main.merge(rob, on='Test', suffixes=('_main', '_robust'))
    df['delta_AUC'] = df['AUC_mean_robust'] - df['AUC_mean_main']
    df['main_in_robust_CI'] = (df['AUC_mean_main'] >= df['CI_lo_robust']) & \
                              (df['AUC_mean_main'] <= df['CI_hi_robust'])
    df['noticeable'] = df['delta_AUC'].abs() > RQ3_AUC_DELTA_FLAG
    df.to_csv(os.path.join(OUT_DIR, f'rq3_combined_diff_{gl}_{sb}.csv'), index=False)
    return df


def write_md(rq1_results, rq2_df, rq3_data):
    md = []
    md.append(f"# Diff-grouping Robustness Check — Comparison ({SCHEME} vs main {MAIN_SCHEME})")
    md.append("")
    md.append(f"> 对比对象: 主分析 ({MAIN_SCHEME}) vs 当前 scheme ({SCHEME})。")
    md.append(f"> 当前 scheme 的 popular cut = 5%, gap cut = 10%, unpopular cut = 85% (top 5% 是 popular, top 5-15% 是 gap, 剩 85% 是 unpopular)。")
    md.append(f"> 阈值: RQ2 |ΔAUC| > {RQ2_AUC_DELTA_FLAG}, RQ3 |ΔAUC| > {RQ3_AUC_DELTA_FLAG} 视为 noticeable。")
    md.append("")

    # RQ1
    md.append("## RQ1 — Significance Analysis")
    md.append("")
    md.append("| sort_by | features | sig (main) | sig (current) | significance flips | effect-size category changes | max \\|Δeffect\\| |")
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
                md.append("| Feature | main | current |")
                md.append("| --- | --- | --- |")
                for c in r['category_changed_features']:
                    md.append(f"| {c['Feature']} | {c['Effect_Size_Interpretation_main']} | "
                              f"{c['Effect_Size_Interpretation_robust']} |")
                md.append("")

    # RQ2
    md.append("## RQ2 — ML Classifiers")
    md.append("")
    md.append("| sort_by | Model | AUC (main) | AUC (current) | ΔAUC | params changed? | noticeable? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in rq2_df.iterrows():
        flag = "**yes**" if row['noticeable_drop'] else "no"
        pc = "yes" if row['params_changed'] else "no"
        md.append(f"| {row['sort_by']} | {row['Model']} | {row['AUC_main']:.4f} | {row['AUC_robust']:.4f} | "
                  f"{row['delta_AUC']:+.4f} | {pc} | {flag} |")
    md.append("")
    md.append(f"**Summary**: {int(rq2_df['noticeable_drop'].sum())}/{len(rq2_df)} pairs ΔAUC 超 {RQ2_AUC_DELTA_FLAG}; "
              f"{int(rq2_df['params_changed'].sum())}/{len(rq2_df)} best params 变化。")
    md.append("")

    # RQ3 - within
    md.append("## RQ3 — Cross-group Analysis")
    md.append("")
    md.append("### Within-group baseline (10-fold CV)")
    md.append("")
    md.append("| sort_by | group_label | group | AUC main | AUC current | ΔAUC | noticeable? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), w in rq3_data['within'].items():
        if w is None: continue
        for _, row in w.iterrows():
            flag = "**yes**" if row['noticeable'] else "no"
            md.append(f"| {sb} | {gl} | {row['Group']} | {row['AUC_mean_main']:.4f} | "
                      f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | {flag} |")
    md.append("")

    # RQ3 - 1-vs-1 summary
    md.append("### Cross-group 1-vs-1 (train A → test B)")
    md.append("")
    md.append("| sort_by | group_label | N cells | mean ΔAUC | max \\|ΔAUC\\| | cells with main outside current CI | noticeable cells |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), s in rq3_data['single'].items():
        if s is None: continue
        md.append(f"| {sb} | {gl} | {len(s)} | {s['delta_AUC'].mean():+.4f} | "
                  f"{s['delta_AUC'].abs().max():.4f} | "
                  f"{int((~s['main_in_robust_CI']).sum())} | "
                  f"{int(s['noticeable'].sum())} |")
    md.append("")
    for (sb, gl), s in rq3_data['single'].items():
        if s is None: continue
        bad = s[s['noticeable']]
        if not bad.empty:
            md.append(f"#### by {sb} / cross-{gl} — noticeable cells ({len(bad)})")
            md.append("")
            md.append("| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            for _, row in bad.iterrows():
                in_ci = "yes" if row['main_in_robust_CI'] else "**no**"
                md.append(f"| {row['Train']} → {row['Test']} | {row['AUC_mean_main']:.4f} | "
                          f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | "
                          f"[{row['CI_lo_robust']:.3f}, {row['CI_hi_robust']:.3f}] | {in_ci} |")
            md.append("")

    # RQ3 - others-vs-1 full table
    md.append("### Cross-group others-vs-1 (others → target)")
    md.append("")
    md.append("| sort_by | group_label | target | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (sb, gl), c in rq3_data['combined'].items():
        if c is None: continue
        for _, row in c.iterrows():
            in_ci = "yes" if row['main_in_robust_CI'] else "**no**"
            md.append(f"| {sb} | {gl} | {row['Test']} | {row['AUC_mean_main']:.4f} | "
                      f"{row['AUC_mean_robust']:.4f} | {row['delta_AUC']:+.4f} | "
                      f"[{row['CI_lo_robust']:.3f}, {row['CI_hi_robust']:.3f}] | {in_ci} |")
    md.append("")

    out_md = os.path.join(script_dir, 'comparison_with_main.md')
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
            s, c = rq3_data['single'][(sb, gl)], rq3_data['combined'][(sb, gl)]
            if s is not None:
                print(f"  {sb}/{gl} 1-vs-1: {len(s)} cells, mean ΔAUC={s['delta_AUC'].mean():+.4f}, "
                      f"max |Δ|={s['delta_AUC'].abs().max():.4f}, "
                      f"main outside CI: {int((~s['main_in_robust_CI']).sum())}, "
                      f"noticeable: {int(s['noticeable'].sum())}")
            if c is not None:
                print(f"  {sb}/{gl} others-vs-1: {len(c)} cells, mean ΔAUC={c['delta_AUC'].mean():+.4f}, "
                      f"main outside CI: {int((~c['main_in_robust_CI']).sum())}")

    write_md(rq1_results, rq2_df, rq3_data)


if __name__ == '__main__':
    main()
