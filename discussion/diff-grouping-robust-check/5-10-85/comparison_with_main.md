# Diff-grouping Robustness Check — Comparison (5-10-85 vs main 10-10-80)

> 对比对象: 主分析 (10-10-80) vs 当前 scheme (5-10-85)。
> 当前 scheme 的 popular cut = 5%, gap cut = 10%, unpopular cut = 85% (top 5% 是 popular, top 5-15% 是 gap, 剩 85% 是 unpopular)。
> 阈值: RQ2 |ΔAUC| > 0.02, RQ3 |ΔAUC| > 0.03 视为 noticeable。

## RQ1 — Significance Analysis

| sort_by | features | sig (main) | sig (current) | significance flips | effect-size category changes | max \|Δeffect\| |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | 31 | 28 | 27 | 5 | 4 | 0.0963 |
| likes | 31 | 29 | 29 | 2 | 1 | 0.0545 |

### by downloads — 变化细节

**Significance flipped (5)**: has_dataset, num_dataset, num_modules, num_model_files, has_space

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| word_count_yaml | small | negligible |
| num_arxiv | small | medium |
| model_size_bytes | negligible | small |
| has_primary_implementation_library_name | small | negligible |

### by likes — 变化细节

**Significance flipped (2)**: has_dataset, has_space

**Effect-size category changed:**

| Feature | main | current |
| --- | --- | --- |
| has_quantized | small | negligible |

## RQ2 — ML Classifiers

| sort_by | Model | AUC (main) | AUC (current) | ΔAUC | params changed? | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | RandomForest | 0.8610 | 0.8874 | +0.0265 | no | **yes** |
| downloads | KNN | 0.8068 | 0.8139 | +0.0071 | no | no |
| downloads | DecisionTree | 0.7798 | 0.7997 | +0.0199 | no | no |
| downloads | SVM | 0.7221 | 0.7625 | +0.0404 | no | **yes** |
| downloads | NaiveBayes | 0.6839 | 0.7127 | +0.0288 | no | **yes** |
| likes | RandomForest | 0.8477 | 0.8719 | +0.0241 | no | **yes** |
| likes | KNN | 0.7823 | 0.7846 | +0.0023 | no | no |
| likes | DecisionTree | 0.7578 | 0.7738 | +0.0160 | yes | no |
| likes | SVM | 0.7243 | 0.7594 | +0.0352 | no | **yes** |
| likes | NaiveBayes | 0.6842 | 0.7112 | +0.0270 | no | **yes** |

**Summary**: 6/10 pairs ΔAUC 超 0.02; 1/10 best params 变化。

## RQ3 — Cross-group Analysis

### Within-group baseline (10-fold CV)

| sort_by | group_label | group | AUC main | AUC current | ΔAUC | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.8143 | 0.7969 | -0.0173 | no |
| downloads | domain | Computer Vision | 0.8910 | 0.8928 | +0.0019 | no |
| downloads | domain | Multimodal | 0.8436 | 0.8490 | +0.0054 | no |
| downloads | domain | NLP | 0.8612 | 0.8839 | +0.0227 | no |
| downloads | domain | Other | 0.8110 | 0.8596 | +0.0486 | **yes** |
| downloads | affiliation | community | 0.8682 | 0.8974 | +0.0292 | no |
| downloads | affiliation | company | 0.8809 | 0.8913 | +0.0104 | no |
| downloads | affiliation | non-profit | 0.8168 | 0.8198 | +0.0029 | no |
| downloads | affiliation | organization or individual | 0.8306 | 0.8514 | +0.0208 | no |
| downloads | affiliation | university | 0.8392 | 0.8629 | +0.0237 | no |
| likes | domain | Audio | 0.8242 | 0.8466 | +0.0224 | no |
| likes | domain | Computer Vision | 0.8517 | 0.8655 | +0.0138 | no |
| likes | domain | Multimodal | 0.8324 | 0.8540 | +0.0216 | no |
| likes | domain | NLP | 0.8484 | 0.8746 | +0.0262 | no |
| likes | domain | Other | 0.8158 | 0.8367 | +0.0209 | no |
| likes | affiliation | community | 0.8646 | 0.8741 | +0.0095 | no |
| likes | affiliation | company | 0.8669 | 0.8867 | +0.0198 | no |
| likes | affiliation | non-profit | 0.8389 | 0.8424 | +0.0035 | no |
| likes | affiliation | organization or individual | 0.8252 | 0.8381 | +0.0129 | no |
| likes | affiliation | university | 0.7655 | 0.7972 | +0.0317 | **yes** |

### Cross-group 1-vs-1 (train A → test B)

| sort_by | group_label | N cells | mean ΔAUC | max \|ΔAUC\| | cells with main outside current CI | noticeable cells |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | 20 | +0.0128 | 0.0684 | 8 | 4 |
| downloads | affiliation | 20 | +0.0231 | 0.0864 | 7 | 7 |
| likes | domain | 20 | +0.0157 | 0.0582 | 9 | 6 |
| likes | affiliation | 20 | +0.0172 | 0.0433 | 4 | 4 |

#### by downloads / cross-domain — noticeable cells (4)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| Audio → Computer Vision | 0.7742 | 0.7386 | -0.0356 | [0.714, 0.763] | **no** |
| Computer Vision → Multimodal | 0.7035 | 0.6609 | -0.0425 | [0.628, 0.695] | **no** |
| Computer Vision → Other | 0.6446 | 0.7130 | +0.0684 | [0.689, 0.738] | **no** |
| NLP → Other | 0.7061 | 0.7726 | +0.0666 | [0.746, 0.799] | **no** |

#### by downloads / cross-affiliation — noticeable cells (7)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| community → organization or individual | 0.6879 | 0.7249 | +0.0370 | [0.708, 0.740] | **no** |
| company → community | 0.6776 | 0.7236 | +0.0460 | [0.695, 0.752] | **no** |
| company → organization or individual | 0.6555 | 0.6906 | +0.0351 | [0.677, 0.705] | **no** |
| organization or individual → community | 0.6865 | 0.7729 | +0.0864 | [0.746, 0.800] | **no** |
| organization or individual → university | 0.7161 | 0.7603 | +0.0441 | [0.731, 0.786] | **no** |
| university → community | 0.6718 | 0.7256 | +0.0538 | [0.693, 0.758] | **no** |
| university → organization or individual | 0.6709 | 0.7059 | +0.0350 | [0.688, 0.722] | **no** |

#### by likes / cross-domain — noticeable cells (6)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| Audio → NLP | 0.6705 | 0.7051 | +0.0346 | [0.694, 0.717] | **no** |
| Computer Vision → Multimodal | 0.6917 | 0.7311 | +0.0394 | [0.699, 0.762] | **no** |
| Computer Vision → NLP | 0.6986 | 0.7331 | +0.0345 | [0.721, 0.745] | **no** |
| Computer Vision → Other | 0.6380 | 0.6962 | +0.0582 | [0.673, 0.716] | **no** |
| Multimodal → Audio | 0.6911 | 0.7220 | +0.0310 | [0.684, 0.761] | yes |
| Multimodal → Other | 0.6806 | 0.6441 | -0.0364 | [0.622, 0.667] | **no** |

#### by likes / cross-affiliation — noticeable cells (4)

| Train → Test | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| community → organization or individual | 0.6911 | 0.7237 | +0.0326 | [0.708, 0.739] | **no** |
| community → university | 0.6346 | 0.6779 | +0.0433 | [0.631, 0.727] | yes |
| company → community | 0.6977 | 0.7372 | +0.0395 | [0.712, 0.764] | **no** |
| university → non-profit | 0.6506 | 0.6832 | +0.0326 | [0.646, 0.720] | yes |

### Cross-group others-vs-1 (others → target)

| sort_by | group_label | target | AUC main | AUC current | ΔAUC | current 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.6895 | 0.7120 | +0.0225 | [0.676, 0.749] | yes |
| downloads | domain | Computer Vision | 0.7756 | 0.7979 | +0.0223 | [0.775, 0.819] | yes |
| downloads | domain | Multimodal | 0.7610 | 0.7679 | +0.0069 | [0.738, 0.797] | yes |
| downloads | domain | NLP | 0.7127 | 0.7336 | +0.0209 | [0.722, 0.745] | **no** |
| downloads | domain | Other | 0.7203 | 0.7836 | +0.0633 | [0.760, 0.808] | **no** |
| downloads | affiliation | community | 0.7277 | 0.7870 | +0.0593 | [0.760, 0.815] | **no** |
| downloads | affiliation | company | 0.7079 | 0.7119 | +0.0040 | [0.698, 0.726] | yes |
| downloads | affiliation | non-profit | 0.7084 | 0.7172 | +0.0088 | [0.684, 0.751] | yes |
| downloads | affiliation | organization or individual | 0.7052 | 0.7448 | +0.0396 | [0.731, 0.758] | **no** |
| downloads | affiliation | university | 0.7008 | 0.7339 | +0.0331 | [0.707, 0.758] | **no** |
| likes | domain | Audio | 0.7815 | 0.7906 | +0.0091 | [0.754, 0.825] | yes |
| likes | domain | Computer Vision | 0.7559 | 0.7710 | +0.0151 | [0.753, 0.789] | yes |
| likes | domain | Multimodal | 0.7687 | 0.7910 | +0.0223 | [0.764, 0.816] | yes |
| likes | domain | NLP | 0.7539 | 0.7832 | +0.0293 | [0.772, 0.794] | **no** |
| likes | domain | Other | 0.7478 | 0.7576 | +0.0099 | [0.738, 0.776] | yes |
| likes | affiliation | community | 0.7164 | 0.7592 | +0.0428 | [0.735, 0.784] | **no** |
| likes | affiliation | company | 0.7067 | 0.7340 | +0.0273 | [0.721, 0.748] | **no** |
| likes | affiliation | non-profit | 0.7223 | 0.7288 | +0.0066 | [0.695, 0.764] | yes |
| likes | affiliation | organization or individual | 0.7165 | 0.7433 | +0.0268 | [0.729, 0.757] | **no** |
| likes | affiliation | university | 0.6449 | 0.6627 | +0.0179 | [0.613, 0.712] | yes |

