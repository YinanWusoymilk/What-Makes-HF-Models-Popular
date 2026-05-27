this paper-->

> 中文工作稿: [`README.zh.md`](README.zh.md)


## Table of Contents

- [Data Collection](#data-collection)
  - [Step 1: Collect the Public Model Listing](#step-1-collect-the-public-model-listing)
  - [Step 2: Collect the Hugging Face Dataset Index](#step-2-collect-the-hugging-face-dataset-index)
  - [Step 3: Collect Per-Model Details](#step-3-collect-per-model-details)
  - [Step 4: Collect Repository File Trees and File Sizes](#step-4-collect-repository-file-trees-and-file-sizes)
  - [Step 5: Collect READMEs and Convert Markdown to HTML](#step-5-collect-readmes-and-convert-markdown-to-html)
  - [Before Rerunning](#before-rerunning)
- [Feature Extraction](#feature-extraction)
  - [Pipeline Overview](#pipeline-overview)
  - [Step 1: Extract Metadata, Platform, and Technique Features from Listing Metadata](#step-1-extract-metadata-platform-and-technique-features-from-listing-metadata)
  - [Step 2: Extract Model-Structure Features from Per-Model Details and File Trees](#step-2-extract-model-structure-features-from-per-model-details-and-file-trees)
  - [Step 3: Extract Documentation Features from README HTML](#step-3-extract-documentation-features-from-readme-html)
  - [Step 4: Merge the Three Feature Tables](#step-4-merge-the-three-feature-tables)
  - [Rerun Feature Extraction](#rerun-feature-extraction)
  - [Scope Notes](#scope-notes)
- [Preprocessing](#preprocessing)
  - [Step 1: Filter and Group](#step-1-filter-and-group)
  - [Step 2: Distribution Visualization](#step-2-distribution-visualization)
  - [Step 3: Manual Sanity Check](#step-3-manual-sanity-check)
  - [Rerun Preprocessing](#rerun-preprocessing)


## Data Collection

This stage collects the raw Hugging Face data used by the later feature
extraction and preprocessing scripts. The relevant scripts live under
[`data-collection/`](data-collection/). They are mainly useful for understanding
how the raw data was collected, rather than as a polished one-command
reproduction entry point. Some scripts depend on a local `config.yaml` file in
which users must provide their own Hugging Face API key; this repository does
not provide or manage private credentials.

### Step 1: Collect the Public Model Listing

[`data-collection/metadata-collection/crawl-metadata.py`](data-collection/metadata-collection/crawl-metadata.py)
requests the Hugging Face model listing endpoint (`/api/models`) page by page.
The core logic is in `HuggingFaceAPI.get_all_models()`: it starts from
`/api/models?limit=1000`, follows `response.links["next"]` for pagination,
sleeps between requests, and accumulates the returned model metadata. The script
saves both JSON and CSV files named with the collection date and the number of
returned entries.

This listing JSON is the root input for later crawlers. In the current pipeline,
downstream scripts read the following model-list file by default:
`2026-02-28_all_huggingface_models_2669509.json`.

### Step 2: Collect the Hugging Face Dataset Index

[`data-collection/dataset-collection/dataset-collection.py`](data-collection/dataset-collection/dataset-collection.py)
requests `/api/datasets` with the same paginated pattern. The main output used
by this project is the plain-text `*_dataset_ids.txt` file, which is used by
later stages.

### Step 3: Collect Per-Model Details

[`data-collection/model-details-collection/crawl-model-details.py`](data-collection/model-details-collection/crawl-model-details.py)
uses the model IDs from the model-list file and requests
`/api/models/{model_id}` for each model. `worker()` writes one JSON object per
line to `model_details.jsonl`, so interrupted runs can resume from existing
progress. `load_processed_ids()` skips completed or unrecoverable entries, and
`merge_jsonl_to_big_json()` merges the JSONL progress log into
`model_details_final.json`.

The script uses `start_index` and `end_index` to choose which slice of models to
collect. Users can adjust these values for their own target range; for example,
a small test run can use:

```python
start_index, end_index = 0, 1000
```

To collect all models, run multiple batches with different ranges.

### Step 4: Collect Repository File Trees and File Sizes

[`data-collection/model-size-collection/crawl-model-size.py`](data-collection/model-size-collection/crawl-model-size.py)
collects each model repository's file tree through
`/api/models/{model_id}/tree/main`. The key logic is divided between
`get_tree_contents()` and `crawl_recursive()`: the first requests one directory
level, and the second recursively enters subdirectories and records file paths
with the byte sizes returned by the API. The JSONL progress file is later merged
into `model_size_tree_final.json`, which is used to compute `model_size_bytes`.

This crawler also uses `start_index` and `end_index` to choose the collection
range. We recommend testing on a small range before expanding it:

```python
start_index, end_index = 0, 1000
```

Because some repository trees are large, the script includes retry handling,
rate-limit backoff, and a per-model timeout.

### Step 5: Collect READMEs and Convert Markdown to HTML

[`data-collection/model-readme-collection/crawl-readme.py`](data-collection/model-readme-collection/crawl-readme.py)
downloads each model card from `{model_id}/raw/main/README.md`. Successful
downloads are saved under `model-readme-collection/readmes/`, and collection
status is recorded in `readme_progress.jsonl`.

Two conversion scripts prepare the README corpus for feature extraction:

- [`convert_md_mistune.py`](data-collection/model-readme-collection/convert_md_mistune.py)
  parses Markdown with Mistune, extracts the YAML header into a `yaml-metadata`
  HTML block, and writes the output to `html_mistune/`.
- [`convert_md_pandoc.py`](data-collection/model-readme-collection/convert_md_pandoc.py)
  converts the same Markdown files with Pandoc and writes the output to
  `html_pandoc/`. The later table-detection feature uses the Pandoc version
  because it handles Markdown tables more reliably.

### Before Rerunning

- Users must create the `config.yaml` files referenced by the crawlers and add
  their own Hugging Face API key. This repository does not provide or manage
  private credentials.
- The model details and model size scripts should be treated as batchable
  crawlers. Before running them, set `start_index` and `end_index` according to
  the intended collection range.


## Feature Extraction

This stage converts the raw Hugging Face data collected in the previous stage
into feature tables used by the later filtering, labelling, and RQ analyses.
The relevant scripts live under
[`feature-extraction/`](feature-extraction/). The outputs of this stage still
cover the deduplicated full model collection; the final 66,909-model dataset is
constructed later by applying the recency, size, likes, and repository-status
filters.

### Pipeline Overview

```
data-collection/                       (raw JSON / README HTML files)
        │
        ▼
metadata_features.py      ─► metadata_features.csv         (26 cols)
model_details_features.py ─► model_detail_features.csv     (17 cols)
readme_features.py        ─► readme_features.csv           (25 cols)
        │
        ▼
merge_all_features.py     ─► full_model_features.csv       (65 cols)
```

Current row and column counts:

| File | Rows | Columns |
| --- | ---: | ---: |
| `metadata_features.csv` | 2,669,500 | 26 |
| `model_detail_features.csv` | 2,669,500 | 17 |
| `readme_features.csv` | 1,663,799 | 25 |
| `full_model_features.csv` | 2,669,500 | 65 |

`readme_features.csv` has fewer rows because it only contains models with
successfully collected and converted README files; the merge step fills missing
numeric README columns with 0.

### Step 1: Extract Metadata, Platform, and Technique Features from Listing Metadata

[`feature-extraction/metadata_features.py`](feature-extraction/metadata_features.py)
reads the model-listing JSON collected earlier:
`data-collection/metadata-collection/2026-02-28_all_huggingface_models_2669509.json`.
The script deduplicates models by Hugging Face model id and writes
[`feature-extraction/metadata_features.csv`](feature-extraction/metadata_features.csv).
The current file has 2,669,500 rows and 26 columns.

This script mainly extracts features from the structured fields and `tags`
returned by the listing API:

- `extract_content(tags, prefix)` extracts values from tags such as `dataset:`,
  `arxiv:`, `license:`, and `region:`.
- `find_dataset_ids_file()` and `read_huggingface_dataset_ids()` load the
  dataset index; `check_dataset_in_huggingface()` checks whether a declared
  dataset tag matches an existing Hugging Face dataset id.
- `check_if_supported()` checks whether `library_name` belongs to the
  `official_libraries` list embedded in the script.
- The quantization logic checks both model ids and tags for keywords such as
  `gguf`, `gptq`, `awq`, `4bit`, `8bit`, and `quantized`, while excluding
  reverse-meaning strings such as `unquantized` and `non-quantized`.

This stage covers paper features related to datasets, arXiv, licenses,
SafeTensors, quantization, Spaces, pipeline tags, primary implementation
libraries, and additional frameworks/libraries.

### Step 2: Extract Model-Structure Features from Per-Model Details and File Trees

[`feature-extraction/model_details_features.py`](feature-extraction/model_details_features.py)
reads two data-collection outputs:

- `data-collection/model-details-collection/model_details_final.json`
- `data-collection/model-size-collection/model_size_tree_final.json`

It writes
[`feature-extraction/model_detail_features.csv`](feature-extraction/model_detail_features.csv),
which currently has 2,669,500 rows and 17 columns.

The script processes information from the per-model endpoint and repository file
tree:

- `has_non_empty_results()` checks whether the YAML `model-index` contains a
  non-empty `results` block, producing `has_model_index_result`.
- The script walks the `siblings` file list to count root-level files,
  top-level modules, and model-weight files ending in `.bin`, `.safetensors`,
  `.pt`, `.pth`, `.ckpt`, `.h5`, `.onnx`, or `.gguf`.
- `calculate_model_size()` sums the per-file byte sizes returned by the file
  tree API to compute `model_size_bytes` without downloading model weights.
- `check_repository()` and `check_if_restricted()` preserve repository
  availability and restricted-status fields for later filtering.

This stage covers features such as `has_config`, `has_model_index_result`,
`model_size_bytes`, `num_root_file`, `num_modules`, `num_model_files`, and
`has_widgetData`.

### Step 3: Extract Documentation Features from README HTML

[`feature-extraction/readme_features.py`](feature-extraction/readme_features.py)
reads the README HTML files generated in the data-collection stage:

- `data-collection/model-readme-collection/html_mistune/*.html`
- `data-collection/model-readme-collection/html_pandoc/*.html`

It writes
[`feature-extraction/readme_features.csv`](feature-extraction/readme_features.csv),
which currently has 1,663,799 rows and 25 columns. This row count is smaller
than the full model collection because only models with successfully collected
and converted READMEs appear in this table; the merge step fills missing README
numeric features with 0.

README features are extracted by traversing HTML with BeautifulSoup:

- `extract_yaml_features()` separates the Mistune-generated `yaml-metadata`
  block and counts words in the YAML block and Markdown body.
- `count_code()` distinguishes code blocks from inline code by checking whether
  a `<code>` element's parent is `<pre>`.
- `count_images()` counts static images and GIFs.
- `check_links()` counts GitHub and Hugging Face links and checks for video URLs.
- `has_bibtex()` first checks for a `language-bibtex` code block, then falls
  back to a BibTeX-like keyword pattern using `author=`, `title=`, and `year=`.
- `count_tables_pandoc()` counts `<table>` elements in the Pandoc HTML output,
  because Pandoc handles Markdown pipe tables more reliably than Mistune.

### Step 4: Merge the Three Feature Tables

[`feature-extraction/merge_all_features.py`](feature-extraction/merge_all_features.py)
merges the three CSV files into
[`feature-extraction/full_model_features.csv`](feature-extraction/full_model_features.csv).
The current merged file has 2,669,500 rows and 65 columns.

Several details matter for the merge:

- `metadata_features.csv` uses `_` instead of `/` in `id`; the details table
  keeps the original Hugging Face `model_id`. The script normalizes details ids
  before merging.
- README filenames look like `{model_id_with_underscore}_README.html`; the
  script removes the `_README` suffix to recover the merge id.
- README-level `has_arxiv` is renamed to `has_arxiv_link_in_readme` to avoid
  confusion with metadata-tag `has_arxiv`. The paper's arXiv feature uses the
  metadata-tag definition.
- Numeric README columns are filled with 0 for models without a collected README.

### Rerun Feature Extraction

```bash
cd feature-extraction
python3 metadata_features.py
python3 model_details_features.py
python3 readme_features.py
python3 merge_all_features.py
```

Before rerunning, make sure the required data-collection outputs exist:

- model listing JSON under `data-collection/metadata-collection/`
- dataset id text file under `data-collection/dataset-collection/`
- `model_details_final.json`
- `model_size_tree_final.json`
- Mistune README HTML directory (`data-collection/model-readme-collection/html_mistune/`)
- Pandoc README HTML directory (`data-collection/model-readme-collection/html_pandoc/`)

### Scope Notes

The feature-extraction scripts produce a wide table, not a narrow table with
only the 31 paper-defined features. The current `full_model_features.csv` has
65 columns, including model ids, `downloads`, `likes`, raw content fields,
timestamps, author fields, filtering/status columns, and intermediate checking
columns. The paper and downstream RQ analyses select 31 paper-defined features
from this wider table.

Two README-length features use different names in the code and in the paper,
but the definitions are the same:

| Internal column | Paper-facing feature name | Meaning |
| --- | --- | --- |
| `word_count_yaml` | `length-yaml` | Word count of the README YAML metadata block |
| `word_count_content` | `length-doc` | Word count of the README Markdown body |


## Preprocessing

Terminology note: the paper has a subsection named ``Data Preparation'' inside
the Data Collection section, where it describes how the crawl was performed.
In this README, **Preprocessing** refers to the later repository stage that
filters the merged feature table, assigns popularity groups, creates plots, and
runs the manual sanity check. This stage is implemented in the historical
directory [`data-preproc-dist-analyze/`](data-preproc-dist-analyze/).

This stage filters the wide feature table produced by feature extraction down
to the final analysis dataset, labels each model as popular group / gap buffer /
unpopular group, generates the popularity distribution plots used in the paper, and runs a
manual sanity check that validates the feature-extraction pipeline. The
relevant scripts live under
[`data-preproc-dist-analyze/`](data-preproc-dist-analyze/).

### Step 1: Filter and Group

[`data-preproc-dist-analyze/data_filter_grouped.py`](data-preproc-dist-analyze/data_filter_grouped.py)
reads [`feature-extraction/full_model_features.csv`](feature-extraction/full_model_features.csv)
and applies the following filters in order, using a data collection date of
2026-02-28:

| Step | Remaining rows |
| --- | ---: |
| Raw merged features (65 cols) | 2,669,500 |
| `created_time` < 2025-09-01 (≥180 days before data collection) | 1,935,844 |
| `model_size_bytes` ≥ 300 MiB (314,572,800 bytes) | 967,002 |
| `likes` ≥ 2 | 66,909 |
| `if_repository` == 1 | 66,909 |
| `if_restricted` == 0 | 66,909 |

The filtered model set has **66,909 models and 65 columns**. When the script
writes the `downloads`- or `likes`-grouped CSV files, it adds a `category`
column, so the `filtered_model_data_by_*_*.csv` outputs have **66 columns**.
The `if_repository` and `if_restricted` filters do not remove additional rows
here because the earlier filters already eliminate the affected models, but
they are kept in the pipeline to make the intent explicit.

The script then sorts the 66,909 models by `downloads` and by `likes`
separately and labels each row with the internal CSV column
`category ∈ {popular, gap, unpopular}`. The main analysis uses the
**10-10-80** scheme: the top 10% by the chosen popularity indicator form the
popular group, the bottom 80% form the unpopular group, and the 10% in between
is a gap buffer that is dropped from the popular-vs-unpopular comparison.
Because grouping is based on sorted row positions rather than unique value
thresholds, ties at a boundary can make adjacent groups share the same boundary
value.

The 10-10-80 scheme looks like:

| Sort key | popular group (top 10%) | gap buffer (middle 10%, dropped) | unpopular group (bottom 80%) |
| --- | --- | --- | --- |
| `downloads` | 6,690 models (≥ 1,640 downloads) | 6,691 models (368–1,639) | 53,528 models (≤ 368) |
| `likes` | 6,690 models (≥ 34 likes) | 6,691 models (14–34) | 53,528 models (≤ 14) |

Outputs (in [`data-preproc-dist-analyze/filtered_data/`](data-preproc-dist-analyze/filtered_data/)):

- `filtered_model_data_by_downloads_10-10-80.csv`
- `filtered_model_data_by_likes_10-10-80.csv`

Three additional schemes (5-10-85, 15-10-75, 20-10-70) are also written
to the same folder and are used by the RQ robustness checks.

### Step 2: Distribution Visualization

Two scripts produce the popularity distribution plots referenced in the paper.
Both use log-scale KDE with median and percentile markers.

- [`raw-data-dis-visualization.py`](data-preproc-dist-analyze/raw-data-dis-visualization.py)
  plots downloads and likes on the deduplicated raw model collection
  (2,669,500 models). Output: `raw_distribution_downloads.png`,
  `raw_distribution_likes.png`.
- [`filtered-data-dis-visualization.py`](data-preproc-dist-analyze/filtered-data-dis-visualization.py)
  plots the same indicators on the filtered 66,909-model set, with
  popular group / gap buffer / unpopular group regions shaded and cutoff
  percentiles annotated.
  The 10-10-80 plots used in the paper are:
  `filtered_plots/filtered_distribution_downloads_10-10-80.png` and
  `filtered_plots/filtered_distribution_likes_10-10-80.png`. The script also
  writes the plots for the other three robustness-check schemes alongside.

### Step 3: Manual Sanity Check

[`data-preproc-dist-analyze/feature_extraction_sanity_check/`](data-preproc-dist-analyze/feature_extraction_sanity_check/)
contains a manual sanity check that validates the feature-extraction pipeline
against the live Hugging Face model pages.
[`sample_models.py`](data-preproc-dist-analyze/feature_extraction_sanity_check/sample_models.py)
draws a stratified random sample of 30 models from the main 10-10-80
`downloads` scheme (15 `popular`, 15 `unpopular`; `random_state=42`) and saves
it to `sampled_models.csv`.

One author then manually inspected every sampled Hugging Face model page and
compared it against the pipeline output for all **31 paper-defined features**
across the four study dimensions (Documentation, Models, Platforms,
Techniques). `downloads` and `likes` are excluded from this check because they
are live counters that drift after the snapshot. Across the 30 × 31 = 930
feature-value comparisons, the manually checked values agreed with the
pipeline output in every cell (100% agreement).

### Rerun Preprocessing

```bash
cd data-preproc-dist-analyze
python3 data_filter_grouped.py
python3 raw-data-dis-visualization.py
python3 filtered-data-dis-visualization.py
python3 feature_extraction_sanity_check/sample_models.py
```

Before rerunning, make sure `feature-extraction/full_model_features.csv` and
`data-collection/metadata-collection/2026-02-28_all_huggingface_models_2669509.json`
already exist.


research question:


rq1:



rq2:


rq3:



Discussion:
