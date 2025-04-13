# Data Preprocess

This part introduces the data preprocessing methods, including text data preprocessing and then index data preprocessing.

## I. Text Data Preprocessing

### 1. Text Formatting

#### Jupyter Notebook

1. Collect data from **the-stack-v2**.
2. Perform file-level de-duplication.
3. Split data based on repository stars and forks:
   - [Sum > 0 (prefixed with `md2`)](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Text-Datasets/viewer/code-the-stack-v2-Jupyter_Notebook-md2_scored_classifier-score_4).
   - [Sum = 0 (prefixed with `md`)](https://huggingface.co/datasets/yulan-team/YuLan-Mini-Text-Datasets/viewer/code-the-stack-v2-Jupyter_Notebook-md_scored_classified-score_5?views%5B%5D=code_the_stack_v2_jupyter_notebook_md_scored_classified_score_5).
4. Add Python version as a Markdown annotation (to enable conditioned learning).
5. Perform line-level de-duplication using a text-distance algorithm to filter out repetitive content, such as progress bars, CSV tables, etc.
6. Randomly format a subset of code snippets using **yapf** with random styles.
7. Convert Markdown blocks, Python blocks, and execution results into Markdown format.
8. Score data using [python-edu-scorer](https://huggingface.co/HuggingFaceTB/python-edu-scorer). 
   - High-scoring examples resemble **tutorial-style notebooks**. 
   - Low-scoring examples resemble **spreadsheet processing**.

#### Python

1. Collect data from **the-stack-v2**, **MNBVC**, **SmolLM**, **StarCoder**, **OpenCoder**.
2. Additionally, synthesize pre-train and sft data using Qwen2.5 series models.
3. Perform file-level de-duplication.
4. Apply text metrics filtering (following **DeepSeek Coder** guidelines).
5. Score data using [python-edu-scorer](https://huggingface.co/HuggingFaceTB/python-edu-scorer).

#### C, C++, Rust, HTML, etc.

1. Collect data from **the-stack-v2**, **MNBVC**, **OpenCoder**, etc.
2. Apply text metrics filtering (following **DeepSeek Coder** guidelines).

> Hugging Face has recently open-sourced their [scorers for other languages](https://huggingface.co/collections/HuggingFaceTB/the-ultimate-collection-of-code-classifiers-67b5aa3eb8994a4b71453005), and readers are encouraged to check them out.

#### Math

1. Collect data from **ProofPile2**, **AutoMathText**, and **OpenWebMath-Pro**.
2. Additionally, retrieve math content from **fineweb-edu** and **dclm** using the [math-classifier](https://huggingface.co/yulan-team/math-classifier), and synthesize pre-train and sft data using Qwen2.5 series models.
3. Perform file-level de-duplication.
4. Score data using [fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier).

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
