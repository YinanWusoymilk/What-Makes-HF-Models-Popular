# RQ3: Cross-group Analysis Summary (scheme 10-10-80)

## 方法学

- **Best classifier from RQ2**: RandomForest, 用 `best_model_<sort_by>_10-10-80.csv` 里存的最优超参
- **特征**: RQ2 step 2 选出的特征 (`selected_features_<sort_by>_10-10-80.csv`, 26 个)
- **分组**:
  - Domain: `pipeline_content` 映射到 6 类 (Audio / Computer Vision / Multimodal / NLP / Tabular / Reinforcement Learning); 未映射的归 'Other' (实验中视情况保留或排除)
  - Affiliation: 解析 HF profile page HTML 拿 `span.capitalize`; 默认 fallback `organization or individual`
- **最小样本阈值**: 一个 group 中 popular 和 unpopular 各 ≥ 30
- **Within-group baseline**: 10-fold CV 在同组内 (AUC ± std)
- **Cross-group 1-vs-1**: train on group A, test on group B (A ≠ B)
- **Cross-group others-vs-1**: train on union of all-other-groups, test on target group
- **AUC 不确定性**: 在 test set 上 bootstrap 重采样 1000 次, 报 mean + 95% CI (2.5%–97.5% percentile)

## by downloads — Cross-domain

### Within-group baseline (5 groups)

| Group | N | AUC (10-fold CV) |
| --- | --- | --- |
| Audio | 2187 | 0.814 ± 0.039 |
| Computer Vision | 7853 | 0.891 ± 0.024 |
| Multimodal | 1913 | 0.844 ± 0.022 |
| NLP | 27809 | 0.861 ± 0.010 |
| Other | 20276 | 0.811 ± 0.016 |

### Cross-group 1-vs-1 AUC matrix (行=train, 列=test)

| Train \ Test | Audio | Computer Vision | Multimodal | NLP | Other |
| --- | --- | --- | --- | --- | --- |
| **Audio** | — | 0.774 | 0.710 | 0.616 | 0.667 |
| **Computer Vision** | 0.698 | — | 0.703 | 0.666 | 0.645 |
| **Multimodal** | 0.683 | 0.750 | — | 0.665 | 0.631 |
| **NLP** | 0.672 | 0.777 | 0.771 | — | 0.706 |
| **Other** | 0.682 | 0.710 | 0.664 | 0.710 | — |

注: 数值是 bootstrap AUC mean, CI 见 `cross_<label>_single_<sort_by>_10-10-80.csv`

### Others-vs-1 (general → specific group)

| Test group | N_test | AUC mean | 95% CI |
| --- | --- | --- | --- |
| Audio | 2187 | 0.690 | [0.659, 0.719] |
| Computer Vision | 7853 | 0.776 | [0.757, 0.793] |
| Multimodal | 1913 | 0.761 | [0.738, 0.784] |
| NLP | 27809 | 0.713 | [0.704, 0.721] |
| Other | 20276 | 0.720 | [0.705, 0.736] |

### 跨组 vs 同组 baseline 对比 (评估泛化能力)

对每个 test group, 列同组 baseline AUC vs 跨组平均 AUC (1-vs-1 平均):

| Test group | Within (baseline) | Cross 1-vs-1 mean | Δ (cross − within) |
| --- | --- | --- | --- |
| Audio | 0.814 | 0.684 | -0.131 |
| Computer Vision | 0.891 | 0.753 | -0.138 |
| Multimodal | 0.844 | 0.712 | -0.132 |
| NLP | 0.861 | 0.664 | -0.197 |
| Other | 0.811 | 0.662 | -0.149 |

## by downloads — Cross-affiliation

### Within-group baseline (5 groups)

| Group | N | AUC (10-fold CV) |
| --- | --- | --- |
| community | 5353 | 0.868 ± 0.021 |
| company | 9700 | 0.881 ± 0.013 |
| non-profit | 2374 | 0.817 ± 0.042 |
| organization or individual | 40168 | 0.831 ± 0.009 |
| university | 2403 | 0.839 ± 0.030 |

### Cross-group 1-vs-1 AUC matrix (行=train, 列=test)

| Train \ Test | community | company | non-profit | organization or individual | university |
| --- | --- | --- | --- | --- | --- |
| **community** | — | 0.696 | 0.663 | 0.688 | 0.640 |
| **company** | 0.678 | — | 0.690 | 0.655 | 0.660 |
| **non-profit** | 0.687 | 0.703 | — | 0.681 | 0.639 |
| **organization or individual** | 0.686 | 0.687 | 0.668 | — | 0.716 |
| **university** | 0.672 | 0.681 | 0.676 | 0.671 | — |

注: 数值是 bootstrap AUC mean, CI 见 `cross_<label>_single_<sort_by>_10-10-80.csv`

### Others-vs-1 (general → specific group)

| Test group | N_test | AUC mean | 95% CI |
| --- | --- | --- | --- |
| community | 5353 | 0.728 | [0.705, 0.748] |
| company | 9700 | 0.708 | [0.697, 0.719] |
| non-profit | 2374 | 0.708 | [0.682, 0.734] |
| organization or individual | 40168 | 0.705 | [0.696, 0.714] |
| university | 2403 | 0.701 | [0.675, 0.724] |

### 跨组 vs 同组 baseline 对比 (评估泛化能力)

对每个 test group, 列同组 baseline AUC vs 跨组平均 AUC (1-vs-1 平均):

| Test group | Within (baseline) | Cross 1-vs-1 mean | Δ (cross − within) |
| --- | --- | --- | --- |
| community | 0.868 | 0.681 | -0.188 |
| company | 0.881 | 0.692 | -0.189 |
| non-profit | 0.817 | 0.674 | -0.143 |
| organization or individual | 0.831 | 0.674 | -0.157 |
| university | 0.839 | 0.664 | -0.175 |

## by likes — Cross-domain

### Within-group baseline (5 groups)

| Group | N | AUC (10-fold CV) |
| --- | --- | --- |
| Audio | 2164 | 0.824 ± 0.033 |
| Computer Vision | 7521 | 0.852 ± 0.017 |
| Multimodal | 1942 | 0.832 ± 0.041 |
| NLP | 28302 | 0.848 ± 0.018 |
| Other | 20103 | 0.816 ± 0.022 |

### Cross-group 1-vs-1 AUC matrix (行=train, 列=test)

| Train \ Test | Audio | Computer Vision | Multimodal | NLP | Other |
| --- | --- | --- | --- | --- | --- |
| **Audio** | — | 0.728 | 0.678 | 0.671 | 0.685 |
| **Computer Vision** | 0.759 | — | 0.692 | 0.699 | 0.638 |
| **Multimodal** | 0.691 | 0.705 | — | 0.725 | 0.681 |
| **NLP** | 0.766 | 0.701 | 0.766 | — | 0.742 |
| **Other** | 0.724 | 0.732 | 0.716 | 0.736 | — |

注: 数值是 bootstrap AUC mean, CI 见 `cross_<label>_single_<sort_by>_10-10-80.csv`

### Others-vs-1 (general → specific group)

| Test group | N_test | AUC mean | 95% CI |
| --- | --- | --- | --- |
| Audio | 2164 | 0.781 | [0.756, 0.807] |
| Computer Vision | 7521 | 0.756 | [0.741, 0.770] |
| Multimodal | 1942 | 0.769 | [0.747, 0.791] |
| NLP | 28302 | 0.754 | [0.745, 0.763] |
| Other | 20103 | 0.748 | [0.735, 0.761] |

### 跨组 vs 同组 baseline 对比 (评估泛化能力)

对每个 test group, 列同组 baseline AUC vs 跨组平均 AUC (1-vs-1 平均):

| Test group | Within (baseline) | Cross 1-vs-1 mean | Δ (cross − within) |
| --- | --- | --- | --- |
| Audio | 0.824 | 0.735 | -0.089 |
| Computer Vision | 0.852 | 0.716 | -0.135 |
| Multimodal | 0.832 | 0.713 | -0.119 |
| NLP | 0.848 | 0.708 | -0.141 |
| Other | 0.816 | 0.686 | -0.130 |

## by likes — Cross-affiliation

### Within-group baseline (5 groups)

| Group | N | AUC (10-fold CV) |
| --- | --- | --- |
| community | 5298 | 0.865 ± 0.025 |
| company | 9510 | 0.867 ± 0.014 |
| non-profit | 2317 | 0.839 ± 0.037 |
| organization or individual | 40472 | 0.825 ± 0.010 |
| university | 2402 | 0.765 ± 0.057 |

### Cross-group 1-vs-1 AUC matrix (行=train, 列=test)

| Train \ Test | community | company | non-profit | organization or individual | university |
| --- | --- | --- | --- | --- | --- |
| **community** | — | 0.697 | 0.715 | 0.691 | 0.635 |
| **company** | 0.698 | — | 0.715 | 0.710 | 0.650 |
| **non-profit** | 0.654 | 0.701 | — | 0.627 | 0.615 |
| **organization or individual** | 0.710 | 0.670 | 0.706 | — | 0.641 |
| **university** | 0.647 | 0.648 | 0.651 | 0.632 | — |

注: 数值是 bootstrap AUC mean, CI 见 `cross_<label>_single_<sort_by>_10-10-80.csv`

### Others-vs-1 (general → specific group)

| Test group | N_test | AUC mean | 95% CI |
| --- | --- | --- | --- |
| community | 5298 | 0.716 | [0.694, 0.738] |
| company | 9510 | 0.707 | [0.695, 0.719] |
| non-profit | 2317 | 0.722 | [0.695, 0.749] |
| organization or individual | 40472 | 0.717 | [0.707, 0.726] |
| university | 2402 | 0.645 | [0.613, 0.677] |

### 跨组 vs 同组 baseline 对比 (评估泛化能力)

对每个 test group, 列同组 baseline AUC vs 跨组平均 AUC (1-vs-1 平均):

| Test group | Within (baseline) | Cross 1-vs-1 mean | Δ (cross − within) |
| --- | --- | --- | --- |
| community | 0.865 | 0.677 | -0.188 |
| company | 0.867 | 0.679 | -0.188 |
| non-profit | 0.839 | 0.697 | -0.142 |
| organization or individual | 0.825 | 0.665 | -0.160 |
| university | 0.765 | 0.635 | -0.130 |

