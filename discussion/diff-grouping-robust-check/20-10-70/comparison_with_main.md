# Diff-grouping Robustness Check — Comparison (20-10-70 vs main 10-10-80)

> 对比对象: 主分析 (10-10-80) vs 当前 scheme (20-10-70)。
> 当前 scheme 的 popular cut = 20%, gap cut = 10%, unpopular cut = 70% (top 20% 是 popular, top 20-30% 是 gap, 剩 70% 是 unpopular)。
> 阈值: RQ2 |ΔAUC| > 0.02, RQ3 |ΔAUC| > 0.03 视为 noticeable。

## RQ1 — Significance Analysis

| sort_by | features | sig (main) | sig (current) | significance flips | effect-size category changes | max \|Δeffect\| |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | 31 | 28 | 29 | 3 | 6 | 0.1552 |
| likes | 31 | 29 | 29 | 4 | 2 | 0.0525 |

### by downloads — 变化细节

**Significance flipped (3)**: num_modules, num_model_files, has_space

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| num_inline_code | small | negligible |
| num_huggingface_links | negligible | small |
| num_root_file | negligible | small |
| has_safetensors | small | negligible |
| has_widgetData | negligible | small |
| if_supported_libraries | negligible | small |

### by likes — 变化细节

**Significance flipped (4)**: has_model_index_result, has_dataset, num_dataset, match_huggingface_dataset

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| has_license | negligible | small |
| has_pipeline_name | negligible | small |

## RQ2 — ML Classifiers

| sort_by | Model | AUC (main) | AUC (current) | ΔAUC | params changed? | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | RandomForest | 0.8610 | 0.8495 | -0.0115 | no | no |
| downloads | KNN | 0.8068 | 0.8040 | -0.0028 | no | no |
| downloads | DecisionTree | 0.7798 | 0.7665 | -0.0133 | no | no |
| downloads | SVM | 0.7221 | 0.7012 | -0.0208 | no | **yes** |
| downloads | NaiveBayes | 0.6839 | 0.6692 | -0.0146 | no | no |
| likes | RandomForest | 0.8477 | 0.8181 | -0.0296 | yes | **yes** |
| likes | KNN | 0.7823 | 0.7689 | -0.0135 | no | no |
| likes | DecisionTree | 0.7578 | 0.7423 | -0.0155 | no | no |
| likes | SVM | 0.7243 | 0.6919 | -0.0323 | no | **yes** |
| likes | NaiveBayes | 0.6842 | 0.6590 | -0.0251 | no | **yes** |

**Summary**: 4/10 pairs ΔAUC 超 0.02; 1/10 best params 变化。

## RQ3 — Cross-group Analysis

### Within-group baseline (10-fold CV)

| sort_by | group_label | group | AUC main | AUC current | ΔAUC | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.8143 | 0.8002 | -0.0141 | no |
| downloads | domain | Computer Vision | 0.8910 | 0.8755 | -0.0155 | no |
| downloads | domain | Multimodal | 0.8436 | 0.8235 | -0.0200 | no |
| downloads | domain | NLP | 0.8612 | 0.8354 | -0.0258 | no |
| downloads | domain | Other | 0.8110 | 0.8463 | +0.0353 | **yes** |
| downloads | affiliation | community | 0.8682 | 0.8530 | -0.0152 | no |
| downloads | affiliation | company | 0.8809 | 0.8620 | -0.0188 | no |
| downloads | affiliation | non-profit | 0.8168 | 0.8400 | +0.0232 | no |
| downloads | affiliation | organization or individual | 0.8306 | 0.8405 | +0.0099 | no |
| downloads | affiliation | university | 0.8392 | 0.8153 | -0.0239 | no |
| likes | domain | Audio | 0.8242 | 0.7864 | -0.0378 | **yes** |
| likes | domain | Computer Vision | 0.8517 | 0.8263 | -0.0254 | no |
| likes | domain | Multimodal | 0.8324 | 0.8236 | -0.0089 | no |
| likes | domain | NLP | 0.8484 | 0.8149 | -0.0335 | **yes** |
| likes | domain | Other | 0.8158 | 0.7919 | -0.0239 | no |
| likes | affiliation | community | 0.8646 | 0.8408 | -0.0238 | no |
| likes | affiliation | company | 0.8669 | 0.8477 | -0.0192 | no |
| likes | affiliation | non-profit | 0.8389 | 0.7946 | -0.0443 | **yes** |
| likes | affiliation | organization or individual | 0.8252 | 0.7941 | -0.0310 | **yes** |
| likes | affiliation | university | 0.7655 | 0.7572 | -0.0083 | no |

### Cross-group 1-vs-1 (train A → test B)

| sort_by | group_label | N cells | mean ΔAUC | max \|ΔAUC\| | cells with main outside current CI | noticeable cells |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | 20 | +0.0079 | 0.0882 | 12 | 6 |
| downloads | affiliation | 20 | -0.0084 | 0.0625 | 5 | 4 |
| likes | domain | 20 | -0.0242 | 0.0945 | 12 | 8 |
| likes | affiliation | 20 | -0.0209 | 0.0522 | 14 | 4 |

#### by downloads / cross-domain — noticeable cells (6)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| Audio → Other | 0.6668 | 0.7196 | +0.0528 | [0.710, 0.730] | **no** |
| Computer Vision → Other | 0.6446 | 0.7328 | +0.0882 | [0.724, 0.742] | **no** |
| Multimodal → Other | 0.6307 | 0.7012 | +0.0705 | [0.692, 0.711] | **no** |
| Other → Computer Vision | 0.7096 | 0.7459 | +0.0362 | [0.730, 0.761] | **no** |
| Other → Multimodal | 0.6638 | 0.6988 | +0.0350 | [0.677, 0.720] | **no** |
| Other → NLP | 0.7098 | 0.6650 | -0.0449 | [0.658, 0.672] | **no** |

#### by downloads / cross-affiliation — noticeable cells (4)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| company → organization or individual | 0.6555 | 0.6879 | +0.0324 | [0.682, 0.694] | **no** |
| non-profit → community | 0.6866 | 0.6544 | -0.0322 | [0.638, 0.671] | **no** |
| organization or individual → university | 0.7161 | 0.6537 | -0.0625 | [0.632, 0.675] | **no** |
| university → organization or individual | 0.6709 | 0.6402 | -0.0307 | [0.633, 0.647] | **no** |

#### by likes / cross-domain — noticeable cells (8)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| Computer Vision → Audio | 0.7595 | 0.7153 | -0.0442 | [0.690, 0.740] | **no** |
| Computer Vision → NLP | 0.6986 | 0.6619 | -0.0367 | [0.655, 0.669] | **no** |
| Computer Vision → Other | 0.6380 | 0.6033 | -0.0346 | [0.594, 0.613] | **no** |
| Multimodal → NLP | 0.7253 | 0.6910 | -0.0343 | [0.684, 0.698] | **no** |
| NLP → Audio | 0.7657 | 0.7082 | -0.0575 | [0.683, 0.735] | **no** |
| NLP → Multimodal | 0.7657 | 0.7342 | -0.0315 | [0.713, 0.755] | **no** |
| Other → Computer Vision | 0.7317 | 0.6371 | -0.0945 | [0.623, 0.651] | **no** |
| Other → NLP | 0.7357 | 0.7039 | -0.0318 | [0.697, 0.711] | **no** |

#### by likes / cross-affiliation — noticeable cells (4)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| company → non-profit | 0.7153 | 0.6808 | -0.0345 | [0.659, 0.703] | **no** |
| company → organization or individual | 0.7098 | 0.6603 | -0.0495 | [0.653, 0.668] | **no** |
| non-profit → community | 0.6537 | 0.6144 | -0.0393 | [0.596, 0.632] | **no** |
| organization or individual → non-profit | 0.7061 | 0.6539 | -0.0522 | [0.630, 0.677] | **no** |

### Cross-group others-vs-1 (others → target)

| sort_by | group_label | target | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.6895 | 0.6910 | +0.0015 | [0.667, 0.716] | yes |
| downloads | domain | Computer Vision | 0.7756 | 0.7890 | +0.0135 | [0.774, 0.803] | yes |
| downloads | domain | Multimodal | 0.7610 | 0.7662 | +0.0052 | [0.745, 0.786] | yes |
| downloads | domain | NLP | 0.7127 | 0.6651 | -0.0476 | [0.659, 0.672] | **no** |
| downloads | domain | Other | 0.7203 | 0.7695 | +0.0492 | [0.761, 0.779] | **no** |
| downloads | affiliation | community | 0.7277 | 0.7000 | -0.0277 | [0.684, 0.717] | **no** |
| downloads | affiliation | company | 0.7079 | 0.7083 | +0.0004 | [0.698, 0.718] | yes |
| downloads | affiliation | non-profit | 0.7084 | 0.6954 | -0.0130 | [0.672, 0.718] | yes |
| downloads | affiliation | organization or individual | 0.7052 | 0.7101 | +0.0049 | [0.704, 0.716] | yes |
| downloads | affiliation | university | 0.7008 | 0.6567 | -0.0441 | [0.635, 0.678] | **no** |
| likes | domain | Audio | 0.7815 | 0.7346 | -0.0469 | [0.713, 0.758] | **no** |
| likes | domain | Computer Vision | 0.7559 | 0.7043 | -0.0516 | [0.690, 0.718] | **no** |
| likes | domain | Multimodal | 0.7687 | 0.7439 | -0.0248 | [0.722, 0.764] | **no** |
| likes | domain | NLP | 0.7539 | 0.7164 | -0.0376 | [0.710, 0.723] | **no** |
| likes | domain | Other | 0.7478 | 0.7402 | -0.0075 | [0.732, 0.749] | yes |
| likes | affiliation | community | 0.7164 | 0.7200 | +0.0036 | [0.704, 0.734] | yes |
| likes | affiliation | company | 0.7067 | 0.6791 | -0.0276 | [0.668, 0.690] | **no** |
| likes | affiliation | non-profit | 0.7223 | 0.6778 | -0.0445 | [0.655, 0.701] | **no** |
| likes | affiliation | organization or individual | 0.7165 | 0.6863 | -0.0302 | [0.679, 0.694] | **no** |
| likes | affiliation | university | 0.6449 | 0.6352 | -0.0097 | [0.610, 0.661] | yes |

## Feature Importance — Ranking Stability

> 比较主分析 (10-10-80) vs 当前 scheme (20-10-70) 的 RF 特征重要性排名。
> MI 在每个 scheme 下选不同特征子集, 这里用 inner join 只比较两边共选的特征 (n_shared)。
> 指标: Spearman 排名相关系数 (1.0 = 完全一致), top-10 特征重合度, 排名 delta。

| sort_by | importance kind | n_shared (main / current) | Spearman ρ | p | top-K overlap | max \|Δrank\| | mean \|Δrank\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | perm | 24 (26 / 26) | +0.9191 | 2.29e-10 | 9/10 | 10 | 2.12 |
| downloads | impurity | 24 (26 / 26) | +0.9661 | 2.01e-14 | 9/10 | 5 | 1.50 |
| likes | perm | 25 (26 / 26) | +0.9631 | 1.33e-14 | 10/10 | 5 | 1.44 |
| likes | impurity | 25 (26 / 26) | +0.9969 | 6.10e-27 | 10/10 | 2 | 0.56 |

**解读**: Spearman ρ 越接近 1.0 + top-K overlap 越接近 n / max |Δrank| 越小, 说明特征排名越稳健。

