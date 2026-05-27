this paper-->

> 中文工作稿: [`README.zh.md`](README.zh.md)


## data collection

This stage collects the raw Hugging Face data used by the later preprocessing
and feature-extraction scripts. The relevant scripts live under
[`data-collection/`](data-collection/). They are mainly useful for understanding
how the raw data was collected, rather than as a polished one-command
reproduction entry point. Some scripts depend on a local `config.yaml` file in
which users must provide their own Hugging Face API key; this repository does
not provide or manage private credentials.

### Step 1: collect the public model listing

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

### Step 2: collect the Hugging Face dataset index

[`data-collection/dataset-collection/dataset-collection.py`](data-collection/dataset-collection/dataset-collection.py)
requests `/api/datasets` with the same paginated pattern. The main output used
by this project is the plain-text `*_dataset_ids.txt` file, which is used by
later stages.

### Step 3: collect per-model details

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

### Step 4: collect repository file trees and file sizes

[`data-collection/model-size-collection/crawl-model-size.py`](data-collection/model-size-collection/crawl-model-size.py)
collects each model repository's file tree through
`/api/models/{model_id}/tree/main`. The key logic is split between
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

### Step 5: collect READMEs and convert Markdown to HTML

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

### Before rerunning

- Users must create the `config.yaml` files referenced by the crawlers and add
  their own Hugging Face API key. This repository does not provide or manage
  private credentials.
- The model details and model size scripts should be treated as batchable
  crawlers. Before running them, set `start_index` and `end_index` according to
  the intended collection range.




data preparation:




feature extraction:




research question:


rq1:



rq2:


rq3:



Discussion:
