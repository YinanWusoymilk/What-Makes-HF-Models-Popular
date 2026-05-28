# RQ2: Feature Importance Summary (scheme 10-10-80)

## Method

- **Step 1 - Correlation filtering**: Compute Spearman correlation across the 31 features. For every pair with |ρ| > 0.7, retain the feature with the higher absolute correlation to the binary popularity label and remove the other feature.
- **Step 2 - Mutual information ranking**: Compute mutual information for the retained features, sweep k=5..n, and select k using Random Forest 5-fold cross-validation AUC.
- **Step 3 - Classifier evaluation**: Compare five classifiers (Random Forest / Decision Tree / Linear SVM / Gaussian Naive Bayes / KNN) using GridSearchCV with **10-fold x 10 repeats = 100 folds**. ROC AUC is the primary metric; accuracy, precision, recall, and F1 are secondary metrics. AUC > 0.7 is treated as the reference threshold for acceptable discrimination.
- **Step 4 - Permutation importance**: For the best model by AUC, compute permutation importance on a stratified 80/20 holdout test set with `permutation_importance(n_repeats=30, scoring='roc_auc')`.

## Sorted by downloads

### Step 1: Correlation filtering

Removed 5 features and retained **26** features.
Removed features: `has_dataset`, `num_dataset`, `num_model_files`, `has_primary_implementation_library_name`, `if_supported_libraries`

### Step 2: Mutual information ranking

Optimal **k = 26** (Random Forest 5-fold CV AUC = 0.8544)

| Rank | Feature | MI Score |
| --- | --- | --- |
| 1 | `model_size_bytes` | 0.0311 |
| 2 | `word_count_content` | 0.0265 |
| 3 | `num_arxiv` | 0.0191 |
| 4 | `num_github_links` | 0.0152 |
| 5 | `num_code_blk` | 0.0149 |
| 6 | `has_pipeline_name` | 0.0149 |
| 7 | `has_bibtex` | 0.0140 |
| 8 | `has_config` | 0.0129 |
| 9 | `word_count_yaml` | 0.0122 |
| 10 | `has_supported_additional_frameworks_Libraries` | 0.0122 |
| 11 | `num_inline_code` | 0.0109 |
| 12 | `num_root_file` | 0.0088 |
| 13 | `has_safetensors` | 0.0087 |
| 14 | `num_table` | 0.0083 |
| 15 | `num_huggingface_links` | 0.0060 |
| 16 | `has_widgetData` | 0.0057 |
| 17 | `has_license` | 0.0054 |
| 18 | `num_lists` | 0.0039 |
| 19 | `num_modules` | 0.0032 |
| 20 | `has_quantized` | 0.0026 |
| 21 | `has_model_index_result` | 0.0010 |
| 22 | `has_space` | 0.0006 |
| 23 | `num_static_img` | 0.0002 |
| 24 | `has_video` | 0.0000 |
| 25 | `num_animated_img` | 0.0000 |
| 26 | `match_huggingface_dataset` | 0.0000 |

### Step 3: Classifier ranking by AUC (primary = AUC, secondary = Acc/P/R/F1)

| Model | AUC | Accuracy | Precision | Recall | F1 | Acceptable (AUC>0.7) |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.8610 ± 0.0072 | 0.9061 ± 0.0026 | 0.6080 ± 0.0168 | 0.4373 ± 0.0206 | 0.5085 ± 0.0168 | ✓ |
| KNN | 0.8068 ± 0.0096 | 0.9051 ± 0.0027 | 0.6179 ± 0.0208 | 0.3835 ± 0.0169 | 0.4731 ± 0.0165 | ✓ |
| DecisionTree | 0.7798 ± 0.0102 | 0.7378 ± 0.0117 | 0.2563 ± 0.0091 | 0.7135 ± 0.0205 | 0.3770 ± 0.0104 | ✓ |
| SVM | 0.7221 ± 0.0102 | 0.6588 ± 0.0064 | 0.1942 ± 0.0053 | 0.6575 ± 0.0177 | 0.2998 ± 0.0079 | ✓ |
| NaiveBayes | 0.6839 ± 0.0098 | 0.1516 ± 0.0200 | 0.1141 ± 0.0023 | 0.9810 ± 0.0058 | 0.2045 ± 0.0037 | ✗ |

### Step 4: Permutation importance on best model = **RandomForest** (AUC=0.8610)

Best params: `{'clf__max_depth': 20, 'clf__n_estimators': 200}`

| Rank | Feature | PermImportance (Δ AUC) |
| --- | --- | --- |
| 1 | `word_count_content` | +0.0346 ± 0.0027 |
| 2 | `model_size_bytes` | +0.0272 ± 0.0023 |
| 3 | `num_github_links` | +0.0244 ± 0.0027 |
| 4 | `has_supported_additional_frameworks_Libraries` | +0.0237 ± 0.0019 |
| 5 | `word_count_yaml` | +0.0217 ± 0.0017 |
| 6 | `num_arxiv` | +0.0186 ± 0.0015 |
| 7 | `num_code_blk` | +0.0133 ± 0.0019 |
| 8 | `num_root_file` | +0.0129 ± 0.0022 |
| 9 | `has_quantized` | +0.0112 ± 0.0011 |
| 10 | `has_config` | +0.0111 ± 0.0010 |
| 11 | `num_huggingface_links` | +0.0108 ± 0.0010 |
| 12 | `has_safetensors` | +0.0105 ± 0.0017 |
| 13 | `num_lists` | +0.0093 ± 0.0010 |
| 14 | `has_pipeline_name` | +0.0082 ± 0.0015 |
| 15 | `num_inline_code` | +0.0072 ± 0.0010 |
| 16 | `match_huggingface_dataset` | +0.0067 ± 0.0012 |
| 17 | `has_license` | +0.0056 ± 0.0009 |
| 18 | `num_modules` | +0.0052 ± 0.0009 |
| 19 | `has_bibtex` | +0.0049 ± 0.0009 |
| 20 | `has_widgetData` | +0.0047 ± 0.0007 |
| 21 | `num_table` | +0.0047 ± 0.0007 |
| 22 | `num_static_img` | +0.0019 ± 0.0008 |
| 23 | `has_model_index_result` | +0.0001 ± 0.0003 |
| 24 | `has_space` | +0.0000 ± 0.0000 |
| 25 | `num_animated_img` | -0.0000 ± 0.0000 |
| 26 | `has_video` | -0.0001 ± 0.0001 |

## Sorted by likes

### Step 1: Correlation filtering

Removed 5 features and retained **26** features.
Removed features: `has_dataset`, `num_dataset`, `num_model_files`, `has_supported_additional_frameworks_Libraries`, `if_supported_libraries`

### Step 2: Mutual information ranking

Optimal **k = 26** (Random Forest 5-fold CV AUC = 0.8413)

| Rank | Feature | MI Score |
| --- | --- | --- |
| 1 | `model_size_bytes` | 0.0268 |
| 2 | `word_count_content` | 0.0248 |
| 3 | `num_arxiv` | 0.0154 |
| 4 | `num_code_blk` | 0.0121 |
| 5 | `num_inline_code` | 0.0110 |
| 6 | `num_github_links` | 0.0109 |
| 7 | `has_bibtex` | 0.0105 |
| 8 | `has_pipeline_name` | 0.0103 |
| 9 | `has_license` | 0.0093 |
| 10 | `has_quantized` | 0.0082 |
| 11 | `num_table` | 0.0072 |
| 12 | `num_huggingface_links` | 0.0070 |
| 13 | `has_config` | 0.0067 |
| 14 | `num_root_file` | 0.0062 |
| 15 | `has_primary_implementation_library_name` | 0.0060 |
| 16 | `word_count_yaml` | 0.0058 |
| 17 | `num_static_img` | 0.0055 |
| 18 | `num_modules` | 0.0052 |
| 19 | `num_lists` | 0.0046 |
| 20 | `has_widgetData` | 0.0045 |
| 21 | `has_safetensors` | 0.0040 |
| 22 | `has_video` | 0.0012 |
| 23 | `has_model_index_result` | 0.0009 |
| 24 | `has_space` | 0.0007 |
| 25 | `num_animated_img` | 0.0000 |
| 26 | `match_huggingface_dataset` | 0.0000 |

### Step 3: Classifier ranking by AUC (primary = AUC, secondary = Acc/P/R/F1)

| Model | AUC | Accuracy | Precision | Recall | F1 | Acceptable (AUC>0.7) |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.8477 ± 0.0064 | 0.9023 ± 0.0024 | 0.6758 ± 0.0329 | 0.2316 ± 0.0157 | 0.3447 ± 0.0197 | ✓ |
| KNN | 0.7823 ± 0.0080 | 0.8976 ± 0.0025 | 0.5900 ± 0.0262 | 0.2583 ± 0.0159 | 0.3591 ± 0.0186 | ✓ |
| DecisionTree | 0.7578 ± 0.0091 | 0.7118 ± 0.0115 | 0.2339 ± 0.0067 | 0.6995 ± 0.0234 | 0.3504 ± 0.0079 | ✓ |
| SVM | 0.7243 ± 0.0089 | 0.6396 ± 0.0085 | 0.1896 ± 0.0052 | 0.6851 ± 0.0180 | 0.2970 ± 0.0075 | ✓ |
| NaiveBayes | 0.6842 ± 0.0099 | 0.3952 ± 0.0561 | 0.1406 ± 0.0070 | 0.8639 ± 0.0392 | 0.2416 ± 0.0091 | ✗ |

### Step 4: Permutation importance on best model = **RandomForest** (AUC=0.8477)

Best params: `{'clf__max_depth': None, 'clf__n_estimators': 200}`

| Rank | Feature | PermImportance (Δ AUC) |
| --- | --- | --- |
| 1 | `word_count_content` | +0.0672 ± 0.0029 |
| 2 | `has_quantized` | +0.0580 ± 0.0032 |
| 3 | `model_size_bytes` | +0.0313 ± 0.0027 |
| 4 | `num_arxiv` | +0.0254 ± 0.0025 |
| 5 | `word_count_yaml` | +0.0198 ± 0.0018 |
| 6 | `num_huggingface_links` | +0.0191 ± 0.0016 |
| 7 | `num_code_blk` | +0.0164 ± 0.0019 |
| 8 | `num_github_links` | +0.0163 ± 0.0023 |
| 9 | `has_license` | +0.0149 ± 0.0014 |
| 10 | `num_root_file` | +0.0121 ± 0.0020 |
| 11 | `num_lists` | +0.0109 ± 0.0014 |
| 12 | `num_modules` | +0.0095 ± 0.0015 |
| 13 | `num_static_img` | +0.0085 ± 0.0011 |
| 14 | `num_inline_code` | +0.0078 ± 0.0014 |
| 15 | `num_table` | +0.0067 ± 0.0011 |
| 16 | `has_config` | +0.0063 ± 0.0011 |
| 17 | `has_pipeline_name` | +0.0062 ± 0.0014 |
| 18 | `has_bibtex` | +0.0054 ± 0.0008 |
| 19 | `has_safetensors` | +0.0052 ± 0.0011 |
| 20 | `match_huggingface_dataset` | +0.0050 ± 0.0010 |
| 21 | `has_primary_implementation_library_name` | +0.0048 ± 0.0011 |
| 22 | `has_widgetData` | +0.0025 ± 0.0009 |
| 23 | `has_model_index_result` | +0.0010 ± 0.0004 |
| 24 | `has_video` | +0.0008 ± 0.0002 |
| 25 | `num_animated_img` | +0.0000 ± 0.0001 |
| 26 | `has_space` | +0.0000 ± 0.0000 |

## Cross-indicator comparison (downloads vs likes)

- **Correlation filtering**: downloads removed ['has_dataset', 'has_primary_implementation_library_name', 'if_supported_libraries', 'num_dataset', 'num_model_files']; likes removed ['has_dataset', 'has_supported_additional_frameworks_Libraries', 'if_supported_libraries', 'num_dataset', 'num_model_files']
- **Removed by both indicators**: ['has_dataset', 'if_supported_libraries', 'num_dataset', 'num_model_files']
- **Selected by MI under both indicators**: ['has_bibtex', 'has_config', 'has_license', 'has_model_index_result', 'has_pipeline_name', 'has_quantized', 'has_safetensors', 'has_space', 'has_video', 'has_widgetData', 'match_huggingface_dataset', 'model_size_bytes', 'num_animated_img', 'num_arxiv', 'num_code_blk', 'num_github_links', 'num_huggingface_links', 'num_inline_code', 'num_lists', 'num_modules', 'num_root_file', 'num_static_img', 'num_table', 'word_count_content', 'word_count_yaml']
- **Permutation-importance top-10 overlap**: ['has_quantized', 'model_size_bytes', 'num_arxiv', 'num_code_blk', 'num_github_links', 'num_root_file', 'word_count_content', 'word_count_yaml']
