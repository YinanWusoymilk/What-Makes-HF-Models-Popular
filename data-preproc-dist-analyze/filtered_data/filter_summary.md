# 数据过滤汇总

## 过滤步骤

| 步骤 | 剩余条数 |
| --- | --- |
| 原始数据 (65 列) | 2669500 |
| created_time < 2025-09-01 (180天前) | 1935844 |
| model_size >= 300 MiB | 967002 |
| likes >= 2 | 66909 |
| if_repository == 1 | 66909 |
| if_restricted == 0 | 66909 |

最终数据: **66909** 条, 65 列

## 分组方案与 category 分布

每格格式: `条数 (值范围)`,值范围对应该组的实际 likes/downloads 区间。

| 排序依据 | 方案 (popular-gap-unpopular) | popular | gap | unpopular | 输出文件 |
| --- | --- | --- | --- | --- | --- |
| downloads | 5-10-85 | 3345 (9531–187393215) | 6691 (716–9523) | 56873 (0–716) | `filtered_model_data_by_downloads_5-10-85.csv` |
| likes | 5-10-85 | 3345 (75–13089) | 6691 (21–75) | 56873 (2–21) | `filtered_model_data_by_likes_5-10-85.csv` |
| downloads | 10-10-80 | 6690 (1640–187393215) | 6691 (368–1639) | 53528 (0–368) | `filtered_model_data_by_downloads_10-10-80.csv` |
| likes | 10-10-80 | 6690 (34–13089) | 6691 (14–34) | 53528 (2–14) | `filtered_model_data_by_likes_10-10-80.csv` |
| downloads | 15-10-75 | 10036 (716–187393215) | 6691 (208–716) | 50182 (0–208) | `filtered_model_data_by_downloads_15-10-75.csv` |
| likes | 15-10-75 | 10036 (21–13089) | 6691 (11–21) | 50182 (2–11) | `filtered_model_data_by_likes_15-10-75.csv` |
| downloads | 20-10-70 | 13381 (368–187393215) | 6691 (128–368) | 46837 (0–128) | `filtered_model_data_by_downloads_20-10-70.csv` |
| likes | 20-10-70 | 13381 (14–13089) | 6691 (8–14) | 46837 (2–8) | `filtered_model_data_by_likes_20-10-70.csv` |
