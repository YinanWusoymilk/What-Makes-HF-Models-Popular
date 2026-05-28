import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu, chi2_contingency
from cliffs_delta import cliffs_delta
from statsmodels.stats.multitest import multipletests

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
SCHEME = '10-10-80'
FILTERED_DIR = os.path.join(script_dir, '../../data-preproc-dist-analyze/filtered_data')

# Analyze the two popularity indicators. Keep downloads first to match the
# preprocessing summaries and figures.
analyses = {
    'downloads': os.path.join(FILTERED_DIR, f'filtered_model_data_by_downloads_{SCHEME}.csv'),
    'likes': os.path.join(FILTERED_DIR, f'filtered_model_data_by_likes_{SCHEME}.csv'),
}

# RR-aligned 31-feature set.
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

# Feature order used in the output tables, aligned with the paper dimensions.
DIMENSION_OF = {}
ordered_features = []
def _add_dim(dim, feats):
    for f in feats:
        DIMENSION_OF[f] = dim
        ordered_features.append(f)

_add_dim('Documentation', [
    'word_count_yaml', 'word_count_content', 'has_video', 'num_code_blk', 'num_inline_code',
    'num_static_img', 'num_animated_img', 'num_lists', 'num_table', 'num_github_links',
    'num_huggingface_links', 'num_arxiv', 'has_bibtex', 'has_license',
])
_add_dim('Models', [
    'has_config', 'has_model_index_result', 'has_dataset', 'num_dataset',
    'match_huggingface_dataset', 'model_size_bytes',
    'num_root_file', 'num_modules', 'num_model_files', 'has_quantized',
])
_add_dim('Platforms', [
    'has_space', 'has_safetensors', 'has_widgetData',
])
_add_dim('Techniques', [
    'has_pipeline_name', 'has_primary_implementation_library_name',
    'has_supported_additional_frameworks_Libraries', 'if_supported_libraries',
])


def compute_cramers_v(table):
    """Compute Cramér's V effect size."""
    stat, p, dof, expected = chi2_contingency(table)
    n = table.sum().sum()
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(stat / (n * min_dim)))


def interpret_cliffs_delta(d):
    abs_d = abs(d)
    if abs_d < 0.147:
        return 'negligible'
    elif abs_d < 0.33:
        return 'small'
    elif abs_d < 0.474:
        return 'medium'
    else:
        return 'large'


def interpret_cramers_v(v):
    if v < 0.1:
        return 'negligible'
    elif v < 0.3:
        return 'small'
    elif v < 0.5:
        return 'medium'
    else:
        return 'large'


def fmt_p(p):
    if p < 1e-300:
        return '<1e-300'
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'


def run_analysis(data_path, sort_by, output_dir, md_lines):
    """Run the RQ1 significance analysis for one popularity indicator."""
    print(f"\n{'='*60}")
    print(f"Analysis: {sort_by} grouping (scheme {SCHEME})")
    print(f"{'='*60}")

    data = pd.read_csv(data_path)
    n_popular = int((data['category'] == 'popular').sum())
    n_unpopular = int((data['category'] == 'unpopular').sum())
    n_gap = int((data['category'] == 'gap').sum())
    print(f"Total models: {len(data)} | popular={n_popular} | unpopular={n_unpopular} | gap (excluded)={n_gap}")

    popular = data[data['category'] == 'popular']
    unpopular = data[data['category'] == 'unpopular']

    results = []

    # --- Continuous features: Mann-Whitney U + Cliff's Delta ---
    print(f"\nAnalyzing {len(features_continuous)} continuous features...")
    for feature in features_continuous:
        if feature not in data.columns:
            print(f"  Warning: skipping {feature} because the column is missing.")
            continue
        x1 = popular[feature].dropna()
        x2 = unpopular[feature].dropna()
        if len(x1) == 0 or len(x2) == 0:
            continue
        stat, p_value = mannwhitneyu(x1, x2, alternative='two-sided')
        d, _ = cliffs_delta(x1, x2)
        results.append({
            'Feature': feature,
            'Dimension': DIMENSION_OF.get(feature, ''),
            'Test_Type': 'Continuous',
            'Test_Applied': 'Mann-Whitney U',
            'p_value': p_value,
            'Effect_Size_Measure': "Cliff's Delta",
            'Effect_Size_Value': d,
            'Effect_Size_Interpretation': interpret_cliffs_delta(d),
            'Popular_Stat': x1.median(),
            'Unpopular_Stat': x2.median(),
            'Stat_Type': 'median',
        })

    # --- Binary features: Chi-square + Cramér's V ---
    print(f"Analyzing {len(features_binary)} binary features...")
    subset = data[data['category'].isin(['popular', 'unpopular'])]
    for feature in features_binary:
        if feature not in data.columns:
            print(f"  Warning: skipping {feature} because the column is missing.")
            continue
        table = pd.crosstab(index=subset[feature], columns=subset['category'])
        if table.shape[0] < 2 or table.shape[1] < 2:
            print(f"  Warning: skipping {feature} because the contingency table is degenerate.")
            continue
        stat, p_value, dof, expected = chi2_contingency(table)
        v = compute_cramers_v(table)
        results.append({
            'Feature': feature,
            'Dimension': DIMENSION_OF.get(feature, ''),
            'Test_Type': 'Binary',
            'Test_Applied': 'Chi-square',
            'p_value': p_value,
            'Effect_Size_Measure': "Cramér's V",
            'Effect_Size_Value': v,
            'Effect_Size_Interpretation': interpret_cramers_v(v),
            'Popular_Stat': popular[feature].mean(),
            'Unpopular_Stat': unpopular[feature].mean(),
            'Stat_Type': 'mean',
        })

    results_df = pd.DataFrame(results)

    # --- Multiple-testing correction (Benjamini-Hochberg FDR) ---
    reject_fdr, pvals_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh', alpha=0.05)
    results_df['p_value_fdr_corrected'] = pvals_fdr
    results_df['significant_after_fdr'] = reject_fdr

    # Save results in the same dimension order used in the paper.
    ordered_results = [results_df[results_df['Feature'] == f].iloc[0]
                       for f in ordered_features
                       if not results_df[results_df['Feature'] == f].empty]
    ordered_df = pd.DataFrame(ordered_results)
    ordered_output = os.path.join(output_dir, f'significance_analysis_by_{sort_by}_{SCHEME}.csv')
    ordered_df.to_csv(ordered_output, index=False)
    print(f"\nSaved ordered results: {ordered_output}")

    # --- Terminal summary ---
    sig_count = int(results_df['significant_after_fdr'].sum())
    total_count = len(results_df)
    print(f"\nSignificance summary after FDR-BH correction (alpha=0.05): {sig_count}/{total_count} significant features")
    print(f"Effect-size distribution: {results_df['Effect_Size_Interpretation'].value_counts().to_dict()}")

    # --- Markdown summary ---
    md_lines.append(f"## {sort_by.capitalize()} grouping (scheme {SCHEME})")
    md_lines.append("")
    md_lines.append(f"- Total models: {len(data)} | popular={n_popular} | unpopular={n_unpopular} | gap (excluded)={n_gap}")
    md_lines.append(f"- Multiple-testing correction: **Benjamini-Hochberg FDR**, alpha=0.05")
    md_lines.append(f"- Significant features after FDR correction: **{sig_count}/{total_count}**")
    md_lines.append(f"- Effect-size distribution: {results_df['Effect_Size_Interpretation'].value_counts().to_dict()}")
    md_lines.append("")

    md_lines.append(f"### Full Results Ordered by Dimension")
    md_lines.append("")
    md_lines.append("> Popular and Unpopular columns report medians for continuous features and proportions (%) for binary features.")
    md_lines.append("")
    md_lines.append("| Dimension | Feature | Test | p (FDR) | Effect | abs(Effect) | Interpretation | Popular | Unpopular | Sig? |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in ordered_df.iterrows():
        sig = '✓' if r['significant_after_fdr'] else ''
        if r['Stat_Type'] == 'mean':
            pop_str = f"{r['Popular_Stat']*100:.1f}%"
            unp_str = f"{r['Unpopular_Stat']*100:.1f}%"
        else:
            pop_str = f"{r['Popular_Stat']:.2f} (median)"
            unp_str = f"{r['Unpopular_Stat']:.2f} (median)"
        md_lines.append(
            f"| {r['Dimension']} | `{r['Feature']}` | {r['Test_Applied']} | "
            f"{fmt_p(r['p_value_fdr_corrected'])} | "
            f"{r['Effect_Size_Measure']}={r['Effect_Size_Value']:.3f} | "
            f"{abs(r['Effect_Size_Value']):.3f} | {r['Effect_Size_Interpretation']} | "
            f"{pop_str} | {unp_str} | {sig} |"
        )
    md_lines.append("")

    # Compact table for features with at least small effect sizes.
    meaningful = ordered_df[ordered_df['Effect_Size_Interpretation'] != 'negligible']
    md_lines.append(f"### Compact Table of Features with at Least Small Effect Sizes")
    md_lines.append("")
    md_lines.append(f"{len(meaningful)}/{len(ordered_df)} features have at least small effect sizes.")
    md_lines.append("")
    md_lines.append("| Dimension | Feature | p (FDR) | abs(Effect) | Interpretation |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for _, r in meaningful.iterrows():
        if not r['significant_after_fdr']:
            continue
        md_lines.append(
            f"| {r['Dimension']} | `{r['Feature']}` | "
            f"{fmt_p(r['p_value_fdr_corrected'])} | "
            f"{abs(r['Effect_Size_Value']):.3f} | "
            f"{r['Effect_Size_Interpretation']} |"
        )
    md_lines.append("")


# --- Main entry point ---
if __name__ == '__main__':
    output_dir = script_dir
    md_lines = [
        f"# RQ1 Significance Analysis Summary (scheme {SCHEME})",
        "",
        "Method:",
        "",
        "- Continuous features: **Mann-Whitney U** + **Cliff's Delta**",
        "- Binary features: **Chi-square** + **Cramér's V**",
        "- Multiple-testing correction: **Benjamini-Hochberg FDR**, alpha=0.05",
        "- The analysis compares only `popular` and `unpopular` models; the `gap` group is excluded as the gap buffer.",
        "",
    ]

    for sort_by, data_path in analyses.items():
        if os.path.exists(data_path):
            run_analysis(data_path, sort_by, output_dir, md_lines)
        else:
            print(f"Warning: file does not exist: {data_path}")

    summary_path = os.path.join(output_dir, f'significance_summary_{SCHEME}.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f"\nSummary written to: {summary_path}")
    print(f"\n{'='*60}\nAll analyses completed.\n{'='*60}")
