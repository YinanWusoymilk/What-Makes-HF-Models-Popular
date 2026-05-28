# RQ1 Significance Analysis Summary (scheme 10-10-80)

Method:

- Continuous features: **Mann-Whitney U** + **Cliff's Delta**
- Binary features: **Chi-square** + **Cramér's V**
- Multiple-testing correction: **Benjamini-Hochberg FDR**, alpha=0.05
- The analysis compares only `popular` and `unpopular` models; the `gap` group is excluded as the gap buffer.

## Downloads grouping (scheme 10-10-80)

- Total models: 66909 | popular=6690 | unpopular=53528 | gap (excluded)=6691
- Multiple-testing correction: **Benjamini-Hochberg FDR**, alpha=0.05
- Significant features after FDR correction: **28/31**
- Effect-size distribution: {'negligible': 18, 'small': 13}

### Full Results Ordered by Dimension

> Popular and Unpopular columns report medians for continuous features and proportions (%) for binary features.

| Dimension | Feature | Test | p (FDR) | Effect | abs(Effect) | Interpretation | Popular | Unpopular | Sig? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Documentation | `word_count_yaml` | Mann-Whitney U | 2.42e-91 | Cliff's Delta=0.152 | 0.152 | small | 16.00 (median) | 13.00 (median) | ✓ |
| Documentation | `word_count_content` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.296 | 0.296 | small | 451.00 (median) | 234.00 (median) | ✓ |
| Documentation | `has_video` | Chi-square | 7.89e-11 | Cramér's V=0.027 | 0.027 | negligible | 2.3% | 1.3% | ✓ |
| Documentation | `num_code_blk` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.282 | 0.282 | small | 2.00 (median) | 1.00 (median) | ✓ |
| Documentation | `num_inline_code` | Mann-Whitney U | 2.57e-165 | Cliff's Delta=0.170 | 0.170 | small | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_static_img` | Mann-Whitney U | 2.61e-15 | Cliff's Delta=-0.045 | 0.045 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_animated_img` | Mann-Whitney U | 0.4700 | Cliff's Delta=0.001 | 0.001 | negligible | 0.00 (median) | 0.00 (median) |  |
| Documentation | `num_lists` | Mann-Whitney U | 1.29e-70 | Cliff's Delta=0.124 | 0.124 | negligible | 1.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_table` | Mann-Whitney U | 1.92e-180 | Cliff's Delta=0.180 | 0.180 | small | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_github_links` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.264 | 0.264 | small | 1.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_huggingface_links` | Mann-Whitney U | 4.45e-40 | Cliff's Delta=0.095 | 0.095 | negligible | 2.00 (median) | 1.00 (median) | ✓ |
| Documentation | `num_arxiv` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.268 | 0.268 | small | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `has_bibtex` | Chi-square | <1e-300 | Cramér's V=0.183 | 0.183 | small | 34.7% | 13.5% | ✓ |
| Documentation | `has_license` | Chi-square | 5.26e-68 | Cramér's V=0.071 | 0.071 | negligible | 84.8% | 75.2% | ✓ |
| Models | `has_config` | Chi-square | 2.95e-176 | Cramér's V=0.116 | 0.116 | small | 79.1% | 61.4% | ✓ |
| Models | `has_model_index_result` | Chi-square | 3.67e-14 | Cramér's V=0.031 | 0.031 | negligible | 7.2% | 5.0% | ✓ |
| Models | `has_dataset` | Chi-square | 0.0067 | Cramér's V=0.011 | 0.011 | negligible | 24.3% | 22.8% | ✓ |
| Models | `num_dataset` | Mann-Whitney U | 0.0011 | Cliff's Delta=0.018 | 0.018 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Models | `match_huggingface_dataset` | Chi-square | 5.85e-09 | Cramér's V=0.024 | 0.024 | negligible | 14.4% | 17.3% | ✓ |
| Models | `model_size_bytes` | Mann-Whitney U | 3.14e-47 | Cliff's Delta=-0.108 | 0.108 | negligible | 6300634922.50 (median) | 12802811387.50 (median) | ✓ |
| Models | `num_root_file` | Mann-Whitney U | 2.19e-41 | Cliff's Delta=0.101 | 0.101 | negligible | 11.00 (median) | 11.00 (median) | ✓ |
| Models | `num_modules` | Mann-Whitney U | 0.2311 | Cliff's Delta=0.007 | 0.007 | negligible | 0.00 (median) | 0.00 (median) |  |
| Models | `num_model_files` | Mann-Whitney U | 0.0652 | Cliff's Delta=-0.014 | 0.014 | negligible | 3.00 (median) | 4.00 (median) |  |
| Models | `has_quantized` | Chi-square | 9.01e-15 | Cramér's V=0.032 | 0.032 | negligible | 23.5% | 28.1% | ✓ |
| Platforms | `has_space` | Chi-square | 0.0485 | Cramér's V=0.008 | 0.008 | negligible | 0.0% | 0.1% | ✓ |
| Platforms | `has_safetensors` | Chi-square | 2.32e-152 | Cramér's V=0.107 | 0.107 | small | 60.2% | 43.2% | ✓ |
| Platforms | `has_widgetData` | Chi-square | 1.37e-75 | Cramér's V=0.075 | 0.075 | negligible | 54.9% | 43.0% | ✓ |
| Techniques | `has_pipeline_name` | Chi-square | 7.21e-216 | Cramér's V=0.128 | 0.128 | small | 83.5% | 64.2% | ✓ |
| Techniques | `has_primary_implementation_library_name` | Chi-square | 1.96e-152 | Cramér's V=0.107 | 0.107 | small | 86.2% | 71.0% | ✓ |
| Techniques | `has_supported_additional_frameworks_Libraries` | Chi-square | 1.33e-194 | Cramér's V=0.121 | 0.121 | small | 81.3% | 62.8% | ✓ |
| Techniques | `if_supported_libraries` | Chi-square | 1.20e-91 | Cramér's V=0.083 | 0.083 | negligible | 81.0% | 68.9% | ✓ |

### Compact Table of Features with at Least Small Effect Sizes

13/31 features have at least small effect sizes.

| Dimension | Feature | p (FDR) | abs(Effect) | Interpretation |
| --- | --- | --- | --- | --- |
| Documentation | `word_count_yaml` | 2.42e-91 | 0.152 | small |
| Documentation | `word_count_content` | <1e-300 | 0.296 | small |
| Documentation | `num_code_blk` | <1e-300 | 0.282 | small |
| Documentation | `num_inline_code` | 2.57e-165 | 0.170 | small |
| Documentation | `num_table` | 1.92e-180 | 0.180 | small |
| Documentation | `num_github_links` | <1e-300 | 0.264 | small |
| Documentation | `num_arxiv` | <1e-300 | 0.268 | small |
| Documentation | `has_bibtex` | <1e-300 | 0.183 | small |
| Models | `has_config` | 2.95e-176 | 0.116 | small |
| Platforms | `has_safetensors` | 2.32e-152 | 0.107 | small |
| Techniques | `has_pipeline_name` | 7.21e-216 | 0.128 | small |
| Techniques | `has_primary_implementation_library_name` | 1.96e-152 | 0.107 | small |
| Techniques | `has_supported_additional_frameworks_Libraries` | 1.33e-194 | 0.121 | small |

## Likes grouping (scheme 10-10-80)

- Total models: 66909 | popular=6690 | unpopular=53528 | gap (excluded)=6691
- Multiple-testing correction: **Benjamini-Hochberg FDR**, alpha=0.05
- Significant features after FDR correction: **29/31**
- Effect-size distribution: {'negligible': 23, 'small': 8}

### Full Results Ordered by Dimension

> Popular and Unpopular columns report medians for continuous features and proportions (%) for binary features.

| Dimension | Feature | Test | p (FDR) | Effect | abs(Effect) | Interpretation | Popular | Unpopular | Sig? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Documentation | `word_count_yaml` | Mann-Whitney U | 1.07e-10 | Cliff's Delta=0.049 | 0.049 | negligible | 14.00 (median) | 13.00 (median) | ✓ |
| Documentation | `word_count_content` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.292 | 0.292 | small | 483.00 (median) | 240.00 (median) | ✓ |
| Documentation | `has_video` | Chi-square | 3.50e-64 | Cramér's V=0.069 | 0.069 | negligible | 3.6% | 1.1% | ✓ |
| Documentation | `num_code_blk` | Mann-Whitney U | 3.10e-270 | Cliff's Delta=0.252 | 0.252 | small | 2.00 (median) | 1.00 (median) | ✓ |
| Documentation | `num_inline_code` | Mann-Whitney U | 5.03e-185 | Cliff's Delta=0.180 | 0.180 | small | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_static_img` | Mann-Whitney U | 4.62e-11 | Cliff's Delta=0.037 | 0.037 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_animated_img` | Mann-Whitney U | 1.58e-11 | Cliff's Delta=0.006 | 0.006 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_lists` | Mann-Whitney U | 3.95e-163 | Cliff's Delta=0.190 | 0.190 | small | 1.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_table` | Mann-Whitney U | 4.81e-76 | Cliff's Delta=0.117 | 0.117 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_github_links` | Mann-Whitney U | 1.16e-254 | Cliff's Delta=0.227 | 0.227 | small | 1.00 (median) | 0.00 (median) | ✓ |
| Documentation | `num_huggingface_links` | Mann-Whitney U | 9.75e-14 | Cliff's Delta=0.054 | 0.054 | negligible | 1.00 (median) | 1.00 (median) | ✓ |
| Documentation | `num_arxiv` | Mann-Whitney U | <1e-300 | Cliff's Delta=0.249 | 0.249 | small | 0.00 (median) | 0.00 (median) | ✓ |
| Documentation | `has_bibtex` | Chi-square | <1e-300 | Cramér's V=0.152 | 0.152 | small | 31.2% | 13.6% | ✓ |
| Documentation | `has_license` | Chi-square | 1.01e-119 | Cramér's V=0.095 | 0.095 | negligible | 87.7% | 74.9% | ✓ |
| Models | `has_config` | Chi-square | 1.09e-56 | Cramér's V=0.065 | 0.065 | negligible | 72.2% | 62.3% | ✓ |
| Models | `has_model_index_result` | Chi-square | 0.0036 | Cramér's V=0.012 | 0.012 | negligible | 6.1% | 5.3% | ✓ |
| Models | `has_dataset` | Chi-square | 0.3037 | Cramér's V=0.004 | 0.004 | negligible | 22.7% | 23.3% |  |
| Models | `num_dataset` | Mann-Whitney U | 0.9118 | Cliff's Delta=-0.001 | 0.001 | negligible | 0.00 (median) | 0.00 (median) |  |
| Models | `match_huggingface_dataset` | Chi-square | 0.0312 | Cramér's V=0.009 | 0.009 | negligible | 16.2% | 17.3% | ✓ |
| Models | `model_size_bytes` | Mann-Whitney U | 1.67e-48 | Cliff's Delta=0.110 | 0.110 | negligible | 15446597121.00 (median) | 11852966123.00 (median) | ✓ |
| Models | `num_root_file` | Mann-Whitney U | 1.97e-32 | Cliff's Delta=0.089 | 0.089 | negligible | 12.00 (median) | 11.00 (median) | ✓ |
| Models | `num_modules` | Mann-Whitney U | 4.68e-87 | Cliff's Delta=0.109 | 0.109 | negligible | 0.00 (median) | 0.00 (median) | ✓ |
| Models | `num_model_files` | Mann-Whitney U | 4.01e-35 | Cliff's Delta=0.092 | 0.092 | negligible | 4.00 (median) | 3.00 (median) | ✓ |
| Models | `has_quantized` | Chi-square | 2.39e-168 | Cramér's V=0.113 | 0.113 | small | 14.7% | 31.1% | ✓ |
| Platforms | `has_space` | Chi-square | 0.0312 | Cramér's V=0.009 | 0.009 | negligible | 0.0% | 0.1% | ✓ |
| Platforms | `has_safetensors` | Chi-square | 3.35e-63 | Cramér's V=0.069 | 0.069 | negligible | 54.3% | 43.4% | ✓ |
| Platforms | `has_widgetData` | Chi-square | 7.00e-32 | Cramér's V=0.048 | 0.048 | negligible | 51.9% | 44.3% | ✓ |
| Techniques | `has_pipeline_name` | Chi-square | 1.94e-128 | Cramér's V=0.098 | 0.098 | negligible | 79.8% | 65.0% | ✓ |
| Techniques | `has_primary_implementation_library_name` | Chi-square | 1.77e-33 | Cramér's V=0.049 | 0.049 | negligible | 79.7% | 72.8% | ✓ |
| Techniques | `has_supported_additional_frameworks_Libraries` | Chi-square | 1.55e-13 | Cramér's V=0.030 | 0.030 | negligible | 70.3% | 65.7% | ✓ |
| Techniques | `if_supported_libraries` | Chi-square | 1.60e-09 | Cramér's V=0.025 | 0.025 | negligible | 74.3% | 70.7% | ✓ |

### Compact Table of Features with at Least Small Effect Sizes

8/31 features have at least small effect sizes.

| Dimension | Feature | p (FDR) | abs(Effect) | Interpretation |
| --- | --- | --- | --- | --- |
| Documentation | `word_count_content` | <1e-300 | 0.292 | small |
| Documentation | `num_code_blk` | 3.10e-270 | 0.252 | small |
| Documentation | `num_inline_code` | 5.03e-185 | 0.180 | small |
| Documentation | `num_lists` | 3.95e-163 | 0.190 | small |
| Documentation | `num_github_links` | 1.16e-254 | 0.227 | small |
| Documentation | `num_arxiv` | <1e-300 | 0.249 | small |
| Documentation | `has_bibtex` | <1e-300 | 0.152 | small |
| Models | `has_quantized` | 2.39e-168 | 0.113 | small |
