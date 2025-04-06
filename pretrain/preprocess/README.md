# Data Preprocess

This part introduces the data preprocessing methods, including text data preprocessing and then index data preprocessing.

## I. Text Data Preprocessing

> We will release the related code soon.

### 1. Text Formatting

For Jupyter Notebook (derived from the-stack-v2), we conduct the following process:

1. File level de-duplication
2. Split by repository stars and forks: [sum>0 split (with prefix md2)](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Text-Datasets/viewer/code-the-stack-v2-Jupyter_Notebook-md2_scored_classifier-score_4) and [sum=0 (with prefix md)](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Text-Datasets/viewer/code-the-stack-v2-Jupyter_Notebook-md_scored_classified-score_5?views%5B%5D=code_the_stack_v2_jupyter_notebook_md_scored_classified_score_5)
3. Add Python version as Markdown annotation (for conditioned learning)
4. Line leve de-duplication using text-distance algorithm to filter data like progress bar, csv table, etc
5. A random number of code snippets will be formatted using yapf with random styles
6. Convert Markdown block, Python block, and execuation results into Markdown format
7. Scoring is done using [python-edu-scorer](https://huggingface.co/HuggingFaceTB/python-edu-scorer) (typical examples: high-scoring code resembles a **tutorial-style notebook**, while low-scoring code looks more like **spreadsheet processing**)

### 2. Data Filtering Pipeline

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

## II. Index Data Preprocessing

This part introduces the process of tokenization, data mixing, packing, etc.

### 1. Preliminary

We store the data before tokenization and after tokenization in two different folders, e.g., `/data/raw` and `/data/input_ids`.

```txt
/data
├── raw
│   ├── dataset-collection-1
│   │   ├── dataset-1
│   │   │   ├── file-1.jsonl
│   │   │   ├── file-2.jsonl
│   │   │   └── ...
│   │   ├── dataset-2
│   │   │   └── file-1.parquet
│   │   └── ...
│   ├── dataset-3
│   └── dataset-collection-2
│       ├── dataset-4
│       │   └── file-1.jsonl
│       └── ...
└── input_ids
    └── dataset-collection-2
        ├── dataset-4
        │   └── file-1
        │       └── wo_ppl
        │           ├── splitted_part-0001.parquet
        │           ├── splitted_part-0001-metadata.json
        │           └── ...
        └── ...
```

The data before tokenization is stored in `parquet` (recommended) or `jsonl` format, which look like:

```json
{"text": "This is a sentence."}
{"text": "This is another sentence."}
```

As for data mixing, we mainly use an online spreadsheet to manage the mixing ratio of different course stages. We provide an [example google spreadsheet](https://docs.google.com/spreadsheets/d/1WJTJuZqSr9kVFqVyNwsOHqvgLwDCjBcW3Pz3d6vwHZs/edit?usp=sharing) for reference.

### 2. Tokenization

The following script will tokenize the files in the target path (including all subfolders), and then split the tokenized data into multiple files according to 0.01B Tokens for more fine-grained data mixing.

```bash
cd tokenize
bash run_tokenize.sh /data/raw/dataset-collection-2/dataset-4
```

> [!NOTE]
> Please modify the script to set:
> 1. default key for text dataset, e.g. "text",
> 2. tokenizer path,
> 3. the save location of the tokenized files,
> 4. context window size, e.g. 4096,
> 5. the number of threads.


### 3. Data Mixing

Now we can mix the data for each curriculum phase. The following steps are required:

1. Update metadata from the google spreadsheet (required only when metadata changes):

```bash
# Copy & paste corresponding columns from spreadsheet following the instruction
cd mix
python update_metadata_from_clipboard.py

# Check the updated metadata
cat datasets.txt
cat subsets.txt
cat sfts.txt
```

2. Generate data mix recipe file:

```bash
# Copy & paste corresponding column from spreadsheet (e.g. column F for phase 1) following the instruction
python mix_from_clipboard.py 1
cat 01_xxxx_xxxx.json  # check the generated recipe file
```

3. Save and pack the mixed data:

```bash
python save_from_recipe.py 01_xxxx_xxxx
```
