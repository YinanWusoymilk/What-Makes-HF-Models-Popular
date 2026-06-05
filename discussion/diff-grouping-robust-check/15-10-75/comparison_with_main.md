# Diff-grouping Robustness Check — Comparison (15-10-75 vs main 10-10-80)

> 对比对象: 主分析 (10-10-80) vs 当前 scheme (15-10-75)。
> 当前 scheme 的 popular cut = 15%, gap cut = 10%, unpopular cut = 75% (top 15% 是 popular, top 15-25% 是 gap, 剩 75% 是 unpopular)。
> 阈值: RQ2 |ΔAUC| > 0.02, RQ3 |ΔAUC| > 0.03 视为 noticeable。

## RQ1 — Significance Analysis

| sort_by | features | sig (main) | sig (current) | significance flips | effect-size category changes | max \|Δeffect\| |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | 31 | 28 | 26 | 6 | 4 | 0.0985 |
| likes | 31 | 29 | 27 | 2 | 2 | 0.0335 |

### by downloads — 变化细节

**Significance flipped (6)**: match_huggingface_dataset, model_size_bytes, num_modules, num_model_files, has_quantized, has_space

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| num_root_file | negligible | small |
| has_safetensors | small | negligible |
| has_widgetData | negligible | small |
| if_supported_libraries | negligible | small |

### by likes — 变化细节

**Significance flipped (2)**: has_model_index_result, match_huggingface_dataset

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| has_license | negligible | small |
| has_pipeline_name | negligible | small |

## RQ2 — ML Classifiers

| sort_by | Model | AUC (main) | AUC (current) | ΔAUC | params changed? | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | RandomForest | 0.8610 | 0.8457 | -0.0153 | no | no |
| downloads | KNN | 0.8068 | 0.7999 | -0.0068 | no | no |
| downloads | DecisionTree | 0.7798 | 0.7687 | -0.0111 | no | no |
| downloads | SVM | 0.7221 | 0.7076 | -0.0145 | no | no |
| downloads | NaiveBayes | 0.6839 | 0.6795 | -0.0044 | no | no |
| likes | RandomForest | 0.8477 | 0.8314 | -0.0163 | no | no |
| likes | KNN | 0.7823 | 0.7781 | -0.0042 | no | no |
| likes | DecisionTree | 0.7578 | 0.7509 | -0.0069 | no | no |
| likes | SVM | 0.7243 | 0.7039 | -0.0204 | no | **yes** |
| likes | NaiveBayes | 0.6842 | 0.6678 | -0.0164 | no | no |

**Summary**: 1/10 pairs ΔAUC 超 0.02; 0/10 best params 变化。

## RQ3 — Cross-group Analysis

### Within-group baseline (10-fold CV)

| sort_by | group_label | group | AUC main | AUC current | ΔAUC | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.8143 | 0.8162 | +0.0019 | no |
| downloads | domain | Computer Vision | 0.8910 | 0.8732 | -0.0178 | no |
| downloads | domain | Multimodal | 0.8436 | 0.8350 | -0.0085 | no |
| downloads | domain | NLP | 0.8612 | 0.8372 | -0.0240 | no |
| downloads | domain | Other | 0.8110 | 0.8052 | -0.0058 | no |
| downloads | affiliation | community | 0.8682 | 0.8553 | -0.0129 | no |
| downloads | affiliation | company | 0.8809 | 0.8712 | -0.0096 | no |
| downloads | affiliation | non-profit | 0.8168 | 0.8399 | +0.0231 | no |
| downloads | affiliation | organization or individual | 0.8306 | 0.8174 | -0.0132 | no |
| downloads | affiliation | university | 0.8392 | 0.8303 | -0.0089 | no |
| likes | domain | Audio | 0.8242 | 0.8066 | -0.0176 | no |
| likes | domain | Computer Vision | 0.8517 | 0.8331 | -0.0186 | no |
| likes | domain | Multimodal | 0.8324 | 0.8336 | +0.0012 | no |
| likes | domain | NLP | 0.8484 | 0.8286 | -0.0199 | no |
| likes | domain | Other | 0.8158 | 0.8089 | -0.0069 | no |
| likes | affiliation | community | 0.8646 | 0.8544 | -0.0102 | no |
| likes | affiliation | company | 0.8669 | 0.8546 | -0.0123 | no |
| likes | affiliation | non-profit | 0.8389 | 0.8187 | -0.0202 | no |
| likes | affiliation | organization or individual | 0.8252 | 0.8060 | -0.0192 | no |
| likes | affiliation | university | 0.7655 | 0.7441 | -0.0213 | no |

### Cross-group 1-vs-1 (train A → test B)

| sort_by | group_label | N cells | mean ΔAUC | max \|ΔAUC\| | cells with main outside current CI | noticeable cells |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | 20 | -0.0014 | 0.0445 | 11 | 5 |
| downloads | affiliation | 20 | -0.0162 | 0.0654 | 9 | 2 |
| likes | domain | 20 | -0.0136 | 0.0527 | 9 | 3 |
| likes | affiliation | 20 | -0.0075 | 0.0480 | 5 | 1 |

#### by downloads / cross-domain — noticeable cells (5)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| Computer Vision → NLP | 0.6656 | 0.6329 | -0.0327 | [0.625, 0.641] | **no** |
| Computer Vision → Other | 0.6446 | 0.6819 | +0.0373 | [0.670, 0.693] | **no** |
| Multimodal → Other | 0.6307 | 0.6717 | +0.0410 | [0.659, 0.683] | **no** |
| Other → Multimodal | 0.6638 | 0.6948 | +0.0310 | [0.669, 0.718] | **no** |
| Other → NLP | 0.7098 | 0.6654 | -0.0445 | [0.657, 0.673] | **no** |

#### by downloads / cross-affiliation — noticeable cells (2)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| organization or individual → company | 0.6872 | 0.6557 | -0.0315 | [0.645, 0.667] | **no** |
| organization or individual → university | 0.7161 | 0.6508 | -0.0654 | [0.628, 0.674] | **no** |

#### by likes / cross-domain — noticeable cells (3)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| NLP → Audio | 0.7657 | 0.7130 | -0.0527 | [0.686, 0.740] | **no** |
| NLP → Computer Vision | 0.7014 | 0.6682 | -0.0332 | [0.652, 0.684] | **no** |
| Other → Computer Vision | 0.7317 | 0.6987 | -0.0330 | [0.684, 0.713] | **no** |

#### by likes / cross-affiliation — noticeable cells (1)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| company → organization or individual | 0.7098 | 0.6619 | -0.0480 | [0.654, 0.670] | **no** |

### Cross-group others-vs-1 (others → target)

| sort_by | group_label | target | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.6895 | 0.6873 | -0.0023 | [0.660, 0.716] | yes |
| downloads | domain | Computer Vision | 0.7756 | 0.7960 | +0.0204 | [0.781, 0.810] | **no** |
| downloads | domain | Multimodal | 0.7610 | 0.7568 | -0.0042 | [0.734, 0.778] | yes |
| downloads | domain | NLP | 0.7127 | 0.6659 | -0.0467 | [0.658, 0.673] | **no** |
| downloads | domain | Other | 0.7203 | 0.7334 | +0.0131 | [0.721, 0.746] | **no** |
| downloads | affiliation | community | 0.7277 | 0.6957 | -0.0320 | [0.677, 0.715] | **no** |
| downloads | affiliation | company | 0.7079 | 0.6835 | -0.0244 | [0.673, 0.694] | **no** |
| downloads | affiliation | non-profit | 0.7084 | 0.6739 | -0.0345 | [0.648, 0.698] | **no** |
| downloads | affiliation | organization or individual | 0.7052 | 0.6826 | -0.0226 | [0.676, 0.690] | **no** |
| downloads | affiliation | university | 0.7008 | 0.6627 | -0.0381 | [0.641, 0.685] | **no** |
| likes | domain | Audio | 0.7815 | 0.7446 | -0.0369 | [0.720, 0.769] | **no** |
| likes | domain | Computer Vision | 0.7559 | 0.7365 | -0.0194 | [0.723, 0.750] | **no** |
| likes | domain | Multimodal | 0.7687 | 0.7549 | -0.0138 | [0.735, 0.777] | yes |
| likes | domain | NLP | 0.7539 | 0.7276 | -0.0264 | [0.720, 0.735] | **no** |
| likes | domain | Other | 0.7478 | 0.7444 | -0.0033 | [0.733, 0.755] | yes |
| likes | affiliation | community | 0.7164 | 0.7101 | -0.0063 | [0.692, 0.728] | yes |
| likes | affiliation | company | 0.7067 | 0.6914 | -0.0153 | [0.682, 0.702] | **no** |
| likes | affiliation | non-profit | 0.7223 | 0.7015 | -0.0207 | [0.677, 0.724] | yes |
| likes | affiliation | organization or individual | 0.7165 | 0.6927 | -0.0238 | [0.685, 0.701] | **no** |
| likes | affiliation | university | 0.6449 | 0.6462 | +0.0013 | [0.619, 0.673] | yes |

## Feature Importance — Ranking Stability

> 比较主分析 (10-10-80) vs 当前 scheme (15-10-75) 的 RF 特征重要性排名。
> MI 在每个 scheme 下选不同特征子集, 这里用 inner join 只比较两边共选的特征 (n_shared)。
> 指标: Spearman 排名相关系数 (1.0 = 完全一致), top-10 特征重合度, 排名 delta。

| sort_by | importance kind | n_shared (main / current) | Spearman ρ | p | top-K overlap | max \|Δrank\| | mean \|Δrank\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | perm | 22 (26 / 24) | +0.9334 | 2.36e-10 | 9/10 | 9 | 1.73 |
| downloads | impurity | 22 (26 / 24) | +0.9718 | 5.17e-14 | 10/10 | 4 | 1.36 |
| likes | perm | 25 (26 / 26) | +0.9823 | 3.09e-18 | 8/10 | 5 | 0.80 |
| likes | impurity | 25 (26 / 26) | +0.9977 | 2.24e-28 | 10/10 | 2 | 0.40 |

**解读**: Spearman ρ 越接近 1.0 + top-K overlap 越接近 n / max |Δrank| 越小, 说明特征排名越稳健。

