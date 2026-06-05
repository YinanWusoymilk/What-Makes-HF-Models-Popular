# Robustness Check vs Main Analysis — Comparison (scheme 10-10-80)

> 对比对象: 主分析 vs 去掉 `downloads` top 1% 后重跑的 robustness analysis。
> 阈值: RQ2 |ΔAUC| > 0.02, RQ3 |ΔAUC| > 0.03 视为 noticeable。

## RQ1 — Significance Analysis

| sort_by | features | sig (main) | sig (robust) | significance flips | effect-size category changes | max \|Δeffect\| |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | 31 | 28 | 27 | 1 | 2 | 0.0456 |
| likes | 31 | 29 | 27 | 2 | 0 | 0.0267 |

### by downloads — 变化细节

**Significance flipped (1)**: has_space

**Effect-size category changed:**

| Feature | main | robust |
| --- | --- | --- |
| has_safetensors | small | negligible |
| has_primary_implementation_library_name | small | negligible |

### by likes — 变化细节

**Significance flipped (2)**: has_model_index_result, match_huggingface_dataset

## RQ2 — ML Classifiers

| sort_by | Model | AUC (main) | AUC (robust) | ΔAUC | params changed? | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | RandomForest | 0.8610 | 0.8467 | -0.0143 | no | no |
| downloads | KNN | 0.8068 | 0.7918 | -0.0149 | no | no |
| downloads | DecisionTree | 0.7798 | 0.7635 | -0.0163 | no | no |
| downloads | SVM | 0.7221 | 0.7071 | -0.0150 | yes | no |
| downloads | NaiveBayes | 0.6839 | 0.6745 | -0.0094 | no | no |
| likes | RandomForest | 0.8477 | 0.8409 | -0.0069 | no | no |
| likes | KNN | 0.7823 | 0.7735 | -0.0088 | no | no |
| likes | DecisionTree | 0.7578 | 0.7518 | -0.0059 | no | no |
| likes | SVM | 0.7243 | 0.7147 | -0.0096 | no | no |
| likes | NaiveBayes | 0.6842 | 0.6773 | -0.0069 | no | no |

**Summary**: 0/10 (model × sort_by) pairs 的 ΔAUC 超过 0.02; 1/10 的 best params 发生变化。

## RQ3 — Cross-group Analysis

### Within-group baseline (10-fold CV)

| sort_by | group_label | group | AUC main | AUC robust | ΔAUC | noticeable? |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.8143 | 0.7914 | -0.0229 | no |
| downloads | domain | Computer Vision | 0.8910 | 0.8779 | -0.0131 | no |
| downloads | domain | Multimodal | 0.8436 | 0.8463 | +0.0027 | no |
| downloads | domain | NLP | 0.8612 | 0.8397 | -0.0214 | no |
| downloads | domain | Other | 0.8110 | 0.8020 | -0.0090 | no |
| downloads | affiliation | community | 0.8682 | 0.8624 | -0.0058 | no |
| downloads | affiliation | company | 0.8809 | 0.8673 | -0.0136 | no |
| downloads | affiliation | non-profit | 0.8168 | 0.8274 | +0.0105 | no |
| downloads | affiliation | organization or individual | 0.8306 | 0.8150 | -0.0156 | no |
| downloads | affiliation | university | 0.8392 | 0.8233 | -0.0159 | no |
| likes | domain | Audio | 0.8242 | 0.8274 | +0.0033 | no |
| likes | domain | Computer Vision | 0.8517 | 0.8468 | -0.0049 | no |
| likes | domain | Multimodal | 0.8324 | 0.8196 | -0.0128 | no |
| likes | domain | NLP | 0.8484 | 0.8366 | -0.0118 | no |
| likes | domain | Other | 0.8158 | 0.8170 | +0.0012 | no |
| likes | affiliation | community | 0.8646 | 0.8562 | -0.0084 | no |
| likes | affiliation | company | 0.8669 | 0.8571 | -0.0098 | no |
| likes | affiliation | non-profit | 0.8389 | 0.8293 | -0.0096 | no |
| likes | affiliation | organization or individual | 0.8252 | 0.8176 | -0.0076 | no |
| likes | affiliation | university | 0.7655 | 0.7525 | -0.0129 | no |

### Cross-group 1-vs-1 (train A → test B)

| sort_by | group_label | N cells | mean ΔAUC | max \|ΔAUC\| | cells with main outside robust CI | noticeable cells |
| --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | 20 | -0.0110 | 0.0418 | 2 | 2 |
| downloads | affiliation | 20 | -0.0174 | 0.0319 | 8 | 1 |
| likes | domain | 20 | -0.0053 | 0.0186 | 2 | 0 |
| likes | affiliation | 20 | -0.0024 | 0.0376 | 6 | 1 |

#### by downloads / cross-domain — noticeable cells (2)

| Train → Test | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| NLP → Other | 0.7061 | 0.6643 | -0.0418 | [0.646, 0.684] | **no** |
| Other → NLP | 0.7098 | 0.6781 | -0.0318 | [0.669, 0.687] | **no** |

#### by downloads / cross-affiliation — noticeable cells (1)

| Train → Test | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| non-profit → company | 0.7031 | 0.6712 | -0.0319 | [0.659, 0.683] | **no** |

#### by likes / cross-affiliation — noticeable cells (1)

| Train → Test | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- |
| non-profit → university | 0.6148 | 0.6524 | +0.0376 | [0.619, 0.685] | **no** |

## Feature Importance — Ranking Stability

> 比较主分析 vs robust 上同一 RF best model 的特征重要性排名。
> 指标: Spearman 排名相关系数 (1.0 = 完全一致), top-10 特征重合度, 排名 delta。

| sort_by | importance kind | n_features | Spearman ρ | p | top-K overlap | max \|Δrank\| | mean \|Δrank\| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | perm | 24 | +0.9722 | 2.35e-15 | 10/10 | 6 | 1.21 |
| downloads | impurity | 24 | +0.9939 | 1.42e-22 | 10/10 | 2 | 0.71 |
| likes | perm | 26 | +0.9850 | 8.21e-20 | 9/10 | 3 | 0.77 |
| likes | impurity | 26 | +0.9925 | 2.08e-23 | 10/10 | 4 | 0.38 |

**解读**: Spearman ρ 越接近 1.0 + top-K overlap 越接近 n / max |Δrank| 越小, 说明特征排名越稳健。

### Cross-group others-vs-1 (others → target)

| sort_by | group_label | target | AUC main | AUC robust | ΔAUC | robust 95% CI | main in CI? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| downloads | domain | Audio | 0.6895 | 0.6894 | -0.0002 | [0.656, 0.720] | yes |
| downloads | domain | Computer Vision | 0.7756 | 0.7749 | -0.0007 | [0.758, 0.792] | yes |
| downloads | domain | Multimodal | 0.7610 | 0.7635 | +0.0024 | [0.739, 0.787] | yes |
| downloads | domain | NLP | 0.7127 | 0.6906 | -0.0220 | [0.682, 0.700] | **no** |
| downloads | domain | Other | 0.7203 | 0.7112 | -0.0092 | [0.696, 0.727] | yes |
| downloads | affiliation | community | 0.7277 | 0.7207 | -0.0071 | [0.699, 0.741] | yes |
| downloads | affiliation | company | 0.7079 | 0.6931 | -0.0148 | [0.681, 0.704] | **no** |
| downloads | affiliation | non-profit | 0.7084 | 0.6835 | -0.0249 | [0.656, 0.710] | yes |
| downloads | affiliation | organization or individual | 0.7052 | 0.6815 | -0.0237 | [0.672, 0.691] | **no** |
| downloads | affiliation | university | 0.7008 | 0.6785 | -0.0223 | [0.651, 0.704] | yes |
| likes | domain | Audio | 0.7815 | 0.7713 | -0.0102 | [0.745, 0.799] | yes |
| likes | domain | Computer Vision | 0.7559 | 0.7443 | -0.0116 | [0.729, 0.758] | yes |
| likes | domain | Multimodal | 0.7687 | 0.7615 | -0.0072 | [0.738, 0.785] | yes |
| likes | domain | NLP | 0.7539 | 0.7480 | -0.0060 | [0.739, 0.757] | yes |
| likes | domain | Other | 0.7478 | 0.7423 | -0.0054 | [0.729, 0.755] | yes |
| likes | affiliation | community | 0.7164 | 0.7177 | +0.0013 | [0.698, 0.739] | yes |
| likes | affiliation | company | 0.7067 | 0.7068 | +0.0001 | [0.696, 0.718] | yes |
| likes | affiliation | non-profit | 0.7223 | 0.7168 | -0.0055 | [0.690, 0.744] | yes |
| likes | affiliation | organization or individual | 0.7165 | 0.7106 | -0.0060 | [0.700, 0.721] | yes |
| likes | affiliation | university | 0.6449 | 0.6536 | +0.0088 | [0.621, 0.687] | yes |

