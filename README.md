<h4 align="center">
    <p>
        <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/README_zh.md">中文</a>| <b>English</b> | <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/README_ja.md">日本語</a>
    <p>
</h4>

<div align=center>
<img src="assets/YuLan-logo.jpg" width="400px">
<h1>YuLan-Mini: An Open Data-efficient Language Model</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/Model_License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2412.17743" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a><img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-Mini"></a>
</div>

YuLan-Mini is a lightweight language model with 2.4 billion parameters. It achieves performance comparable to industry-leading models trained on significantly more data, despite being pre-trained on only 1.08T tokens. The model excels particularly in the domains of **mathematics** and **code**. To facilitate reproducibility, we open-source the relevant [pre-training resources](https://github.com/RUC-GSAI/YuLan-Mini#pre-training-resources-) and [post-training technical report](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/post_train).

---

## News

- [2025.04.11] A clearer data mixture [table](https://docs.google.com/spreadsheets/d/1YP8-loVUxgxo36UEpOwflR3GRHLieBnLlCy8g10g8RU/edit?gid=0#gid=0).
- [2025.03.16] Math, code, & reasoninig classifiers released
- [2025.03.07] W&B Logs for ablation studies released
- [2025.02.28] [YuLan-Mini-Instruct](/post_train) released
- [2025.01.29] YuLan-Mini-Instruct-v1 released
- [2024.12.23] YuLan-Mini & pre-training resources released

## Model Downloads 🔗

> YuLan-Mini is part of the [YuLan family](https://github.com/RUC-GSAI/YuLan-Chat), which includes models with larger sizes and different training strategies.

|  Model  | Context Length | SFT | 🤗 Hugging Face | ModelScope | Wise Model |
|---------|----------------|-----|-----------------|------------|------------|
| YuLan-Mini | 28K | ❎ | [`Base`](https://huggingface.co/yulan-team/YuLan-Mini) | [`Base`](https://modelscope.cn/models/yulan-team/YuLan-Mini) | [`Base`](https://wisemodel.cn/models/yulan-team/YuLan-Mini) |
| YuLan-Mini-Instruct | 28K | ✅ | [`Instruct`](https://huggingface.co/yulan-team/YuLan-Mini-Instruct) | | |

> The intermediate checkpoint can be found [here](https://github.com/RUC-GSAI/YuLan-Mini#pre-training-resources-)

---

## Features 🌟

<div align=center>
<img src="assets/main.png">
</div>

Our pre-training methodology improves training efficiency through three key innovations:

1. an elaborately designed **data pipeline** that combines data cleaning with data schedule strategies;
2. a systematic **optimization method** that can effectively mitigate training instability;
3. an effective **annealing approach** that integrate targeted data selection and long context training.


---
## Behchmarks 🌟

| Models                  | MMLU | CEVAL | GSM8K | ARC_CHALLENGE | GPQA | MATH | HUMANEVAL@1 | MBPP@10 |
|-------------------------|-------|-------|-------|---------------|------|------|-------------|--------|
| Qwen-2.5-1.5B-Instruct  | 57.5  | 65.4  | 73.2  | 47.8          | 29.8 | 55.2 | 61.6        | 88.1   |
| Llama3.2-3B-Instruct    | 60    | 45.9  | 43.4  | 78.6          | 38.6 | 48   | 51.5        | 80.4   |
| YuLan-Mini-Instruct  | 53.6  | 50.5    | 82.3  | 51.8          | 30.1 | 55.2 | 67.7        | 85.7   |


> Note: The model size calculation includes the embedding size.

|      Models      | Model Size | # Train Tokens | Context Length | MATH 500 | GSM 8K | Human Eval | MBPP   | RACE Middle | RACE High | RULER  |
|:----------------|----------:|--------------:|--------------:|:--------|:------|:----------|:------|:-----------|:---------|:------|
|     MiniCPM      |    2.71B    |     1.06T      |       4K       |   15.00  |  53.83 |     50.00* |  47.31 |     56.61   |   44.27   |   N/A  |
|      Qwen-2      |    1.54B    |       7T       |      128K      |   22.60  | 46.90* |     34.80* | 46.90* |     55.77   |   43.69   |  60.16 |
|     Qwen2.5      |    0.49B    |      18T       |      128K      |   23.60  | 41.60* |     30.50* | 39.30* |     52.36   |   40.31   |  49.23 |
|     Qwen2.5      |    1.54B    |      18T       |      128K      |   **45.40**  | **68.50\*** |     37.20* | 60.20* |     **58.77**   |   44.33   |  <ins>68.26</ins> |
|     Gemma2       |    2.61B    |       2T       |       8K       |   18.30* | 30.30* |     19.50* | 42.10* |       -     |      -    |   N/A  |
|    StableLM2     |    1.64B    |       2T       |       4K       |     -    |  20.62 |      8.50* |  17.50 |     56.33   |   **45.06**   |   N/A  |
|    SmolLM2       |    1.71B    |      11T       |       8K       |   11.80  |    -   |     23.35  |  45.00 |     55.77   |   43.06   |   N/A  |
|    Llama3.2      |    3.21B    |       9T       |      128K      |    7.40  |    -   |     29.30  |  49.70 |     55.29   |   43.34   |  **77.06** |
|    YuLan-Mini    |    2.42B    |     1.04T      |       4K       |   32.60  |  66.65 |     <ins>61.60</ins>  |  **66.70** |     55.71   |   43.58   |   N/A  |
|    YuLan-Mini    |    2.42B    |     1.08T      |      28K       |  <ins>37.80</ins>  |  <ins>68.46</ins> |    **64.00**  |  <ins>65.90</ins>|     <ins>57.18</ins>   |   <ins>44.57</ins>   |  51.48 |


|      Models      | LAMBADA | MMLU  | CMMLU | CEval | HellaSwag | WinoGrande | StoryCloze | ARC-e | ARC-c |
|:----------------|:-------|:-----|:-----|:-----|:----------|:-----------|:-----------|:-----|:-----|
|   MiniCPM-2.71B   |  61.91  | 53.37 | 48.97 | 48.24 |   67.92    |     65.74   |     78.51   | 55.51 | 43.86 |
|   Qwen2-1.54B     |  64.68  | 55.90 | **70.76** | **71.94** |   66.11    |     66.14   |     77.60   | 62.21 | 42.92 |
|  Qwen2.5-0.49B    |  52.00  | 47.50 | 52.17 | 54.27 |   50.54    |     55.88   |     71.67   | 56.10 | 39.51 |
|  Qwen2.5-1.54B    |  62.12  | <ins>60.71</ins> | <ins>67.82</ins> | <ins>69.05</ins> |   67.18    |     64.48   |     76.80   | **71.51** | <ins>53.41</ins> |
|   Gemma2-2.61B    |    -    | 52.20*|   -   | 28.00*|   <ins>74.60*</ins>   |    **71.50\***   |       -     |   -   | **55.70\***|
| StableLM2-1.64B   |  66.15  | 40.37 | 29.29 | 26.99 |   69.79    |     64.64   |     <ins>78.56</ins>   | 54.00 | 40.78 |
|  SmolLM2-1.71B    |  <ins>67.42</ins>  | 51.91 | 33.46 | 35.10 |   72.96    |     67.40   |     **79.32**   | 44.82 | 35.49 |
|   Llama3.2-3.21B    |  **69.08**  | **63.40** | 44.44 | 44.49 |   **75.62**    |     <ins>67.48</ins>   |     76.80   | <ins>70.12</ins> | 48.81 |
|    YuLan-Mini-2.42B-4K    |  64.72  | 51.79 | 48.35 | 51.47 |   68.65    |     67.09   |     76.37   | 69.87 | 50.51 |
|    YuLan-Mini-2.42B-28K    |  65.67  | 49.10 | 45.45 | 48.23 |   67.22    |     67.24   |     75.89   | 67.47 | 49.32 |


---

## Pre-Training Resources 🔧

To enhance research transparency and reproducibility, we are open-sourcing relevant [pre-training resources](https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain):

### Pre-Training

<details><summary>1. Pre-training and Evaluation Code</summary>

The pre-training code can be found [here](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain). Note that due to subsequent code modifications, this code may not run directly and may require some adjustments.

<h4 id="step-1-modify-the-config-json-">Step 1: Modify the <code>config.json</code></h4>
<p>Due to the implementation of Hugging Face Trainer, certain parameters are stored in the <code>config.json</code> file and cannot be modified through the Trainer&#39;s command-line arguments. Therefore, you need to update these parameters in the <code>config.json</code> file first, particularly:</p>
<ul>
<li><strong><code>save_steps</code></strong>: The frequency of saving intermediate checkpoints.</li>
<li><strong><code>train_batch_size</code></strong>: The batch size per GPU (equivalent to <code>per_device_train_batch_size</code> in the Trainer). We used a batch size of 1008 (approximately 4M tokens) during the stable training stage. Maintaining this same batch size is equally important for training effectiveness.</li>
</ul>
<p>Below is an example of a properly configured <code>config.json</code> file:</p>
<pre><code class="lang-json">{
  <span class="hljs-attr">"best_metric"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"best_model_checkpoint"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"epoch"</span>: <span class="hljs-number">0.0</span>,
  <span class="hljs-attr">"eval_steps"</span>: <span class="hljs-number">500</span>,
  <span class="hljs-attr">"global_step"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"is_hyper_param_search"</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">"is_local_process_zero"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"is_world_process_zero"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"log_history"</span>: [],
  <span class="hljs-attr">"logging_steps"</span>: <span class="hljs-number">3</span>,
  <span class="hljs-attr">"max_steps"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"num_input_tokens_seen"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"num_train_epochs"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"save_steps"</span>: <span class="hljs-number">250</span>,
  <span class="hljs-attr">"stateful_callbacks"</span>: {
    <span class="hljs-attr">"TrainerControl"</span>: {
      <span class="hljs-attr">"args"</span>: {
        <span class="hljs-attr">"should_epoch_stop"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_evaluate"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_log"</span>: <span class="hljs-literal">false</span>,
        <span class="hljs-attr">"should_save"</span>: <span class="hljs-literal">true</span>,
        <span class="hljs-attr">"should_training_stop"</span>: <span class="hljs-literal">true</span>
      },
      <span class="hljs-attr">"attributes"</span>: {}
    }
  },
  <span class="hljs-attr">"total_flos"</span>: <span class="hljs-number">0</span>,
  <span class="hljs-attr">"train_batch_size"</span>: <span class="hljs-number">3</span>,
  <span class="hljs-attr">"trial_name"</span>: <span class="hljs-literal">null</span>,
  <span class="hljs-attr">"trial_params"</span>: <span class="hljs-literal">null</span>
}
</code></pre>
<h4 id="step-2-enable-universal-checkpointing-in-the-deepspeed-configuration">Step 2: Enable Universal Checkpointing in the DeepSpeed Configuration</h4>
<p>To ensure DeepSpeed Integration loads the Universal Checkpoint, you need to enable this feature in the DeepSpeed configuration JSON file. </p>
<p>Here is an example of a ZeRO2 configuration with Universal Checkpointing enabled:</p>
<pre><code class="lang-json">{
  <span class="hljs-attr">"bf16"</span>: {
    <span class="hljs-attr">"enabled"</span>: <span class="hljs-string">"auto"</span>
  },
  <span class="hljs-attr">"zero_optimization"</span>: {
    <span class="hljs-attr">"stage"</span>: <span class="hljs-number">2</span>,
    <span class="hljs-attr">"allgather_partitions"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"allgather_bucket_size"</span>: <span class="hljs-number">8e8</span>,
    <span class="hljs-attr">"overlap_comm"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"reduce_scatter"</span>: <span class="hljs-literal">true</span>,
    <span class="hljs-attr">"reduce_bucket_size"</span>: <span class="hljs-number">8e8</span>,
    <span class="hljs-attr">"contiguous_gradients"</span>: <span class="hljs-literal">true</span>
  },
  <span class="hljs-attr">"gradient_accumulation_steps"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"gradient_clipping"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"steps_per_print"</span>: <span class="hljs-number">16</span>,
  <span class="hljs-attr">"train_batch_size"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"train_micro_batch_size_per_gpu"</span>: <span class="hljs-string">"auto"</span>,
  <span class="hljs-attr">"wall_clock_breakdown"</span>: <span class="hljs-literal">false</span>,
  <span class="hljs-attr">"dump_state"</span>: <span class="hljs-literal">true</span>,
  <span class="hljs-attr">"optimizer"</span>: {
    <span class="hljs-attr">"type"</span>: <span class="hljs-string">"AdamW"</span>,
    <span class="hljs-attr">"params"</span>: {
      <span class="hljs-attr">"lr"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"betas"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"eps"</span>: <span class="hljs-string">"auto"</span>,
      <span class="hljs-attr">"weight_decay"</span>: <span class="hljs-string">"auto"</span>
    }
  },
  <span class="hljs-attr">"checkpoint"</span>: {
    <span class="hljs-attr">"load_universal"</span>: <span class="hljs-literal">true</span>
  }
}
</code></pre>
<h4 id="step-3-resume-training">Step 3: Resume Training</h4>
<p>When calling <code>trainer.train</code>, include the <code>resume_from_checkpoint</code> argument to load the distributed optimizer state from the Universal Checkpoint and resume training.</p>
<pre><code class="lang-python"><span class="hljs-attr">trainer.train(resume_from_checkpoint</span>=<span class="hljs-string">training_args.resume_from_checkpoint)</span>
</code></pre>
<p>We provide an internal <a href="https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain">training framework</a> for your reference, but you are free to choose other frameworks.</p>

</details>

<details><summary>2. Intermediate Stage Checkpoints</summary>
The intermediate stage checkpoints are released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini</a>.

<table>
    <thead>
        <tr>
            <th>Stage</th>
            <th>Curriculum Phase</th>
            <th>4K Context</th>
            <th>28K Context</th>
            <th>Optimizer</th>
            <th>Inference Architecture</th>
            <th>LAMBADA <code>Acc</code></th>
            <th>GSM8K <code>Acc</code></th>
            <th>HumanEval <code>pass@1</code></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Stable</td>
            <td>5</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase5">YuLan-Mini-Phase5</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>53.85</td>
            <td>3.41</td>
            <td>12.26</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>10</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase10">YuLan-Mini-Phase10</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>55.00</td>
            <td>9.57</td>
            <td>15.95</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>15</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase15">YuLan-Mini-Phase15</a></td>
            <td></td>
            <td></td>
            <td><code>yulanmini</code></td>
            <td>55.81</td>
            <td>13.81</td>
            <td>16.99</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>20</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Phase20">YuLan-Mini-Phase20</a></td>
            <td></td>
            <td>✅</td>
            <td><code>yulanmini</code></td>
            <td>55.81</td>
            <td>21.39</td>
            <td>20.79</td>
        </tr>
        <tr>
            <td>Stable</td>
            <td>25 (1T tokens)</td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">YuLan-Mini-Before-Annealing</a></td>
            <td></td>
            <td>✅</td>
            <td><code>yulanmini</code></td>
            <td>55.67</td>
            <td>29.94</td>
            <td>34.06</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>26</td>
            <td>YuLan-Mini-4K</td>
            <td></td>
            <td></td>
            <td><code>llama</code>*</td>
            <td>64.72</td>
            <td>66.65</td>
            <td>61.60</td>
        </tr>
        <tr>
            <td>Annealing</td>
            <td>27</td>
            <td></td>
            <td><a href="https://huggingface.co/yulan-team/YuLan-Mini">YuLan-Mini</a></td>
            <td></td>
            <td><code>llama</code>*</td>
            <td>65.67</td>
            <td>68.46</td>
            <td>64.00</td>
        </tr>
    </tbody>
</table>

\*: For easier inference and deployment, we merged the re-parameterized added parameters and scaling factors into the final released models ([**YuLan-Mini**](https://huggingface.co/yulan-team/YuLan-Mini) and **YuLan-Mini-Intermediate-4K**), enabling it to run on the Llama architecture. However, these parameters are still retained in the intermediate checkpoints from the training process.

</details>

<details><summary>3. Optimizer States Before Annealing</summary>

<a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">🤗 YuLan-Mini-Before-Annealing</a>
</details>

<details><summary>4. Logs of Ablation Studies</summary>

We provide W&B logs, including the intermediate hidden states and weights of each module, for ablation studies:

<a href="https://wandb.ai/yiwen_hu/YuLan-Mini?nw=dld21gv708t"><img alt="W&B Logs" src="https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black"></a>

<ul>
    <li>WeSaR Re-Param</li>
    <li>Cerebras muP</li>
    <li>Embedding LayerNorm</li>
    <li>Tie Embedding</li>
    <li>QK LayerNorm</li>
    <li>Weight Decay</li>
</ul>
</details>


### Datasets


<details><summary>5. The Used Open-Source Datasets </summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets">Used-Datasets-List</a>

</details>

<details><summary>6. Data Distribution for every phase</summary>

⬇️ Click for more details:
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets/final.pdf">
  <div align=center>
    <img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data_distribution_for_every_phase.png">
  </div>
</a>

</details>

<details><summary>7. Synthetic Data</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/preprocess">Data cleaning</a> and <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/synthesis">synthesis</a> pipeline:

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

The synthetic data we are using is released in <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">🤗 YuLan-Mini-Datasets</a>

Classifiers: <a href="https://huggingface.co/collections/yulan-team/yulan-mini-resources-67b29e0086256b72099b7add">🤗 Yulan-Mini Resources</a>

</details>


### What you can do with these pre-training resources

1. **Pre-train** your own LLM. You can use [our data](https://huggingface.co/yulan-team/YuLan-Mini-Datasets) and curriculum to train a model that's just as powerful as YuLan-Mini.
2. Perform your own **learning rate annealing**. During the annealing phase, YuLan-Mini's learning ability is at its peak. You can resume training from [the checkpoint before annealing](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing) and use your own dataset for learning rate annealing.
3. **Fine-tune** the Instruct version of the LLM. You can use the [YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini) base model to train your own Instruct version.
4. **Training dynamics** research. You can use YuLan-Mini's [intermediate checkpoints](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3) to explore internal changes during the pre-training process.
5. **Synthesize** your own data. You can use YuLan-Mini's [data pipeline](https://github.com/RUC-GSAI/YuLan-Mini) to clean and generate your own dataset.

---

## Quick Start 💻

Below is a simple example for inference using Huggingface:

**Huggingface Inference Example**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yulan-team/YuLan-Mini-Instruct")
model = AutoModelForCausalLM.from_pretrained("yulan-team/YuLan-Mini-Instruct", torch_dtype=torch.bfloat16)

# Input text
chat = [
    {"role": "system", "content": "You are YuLan-Mini, created by RUC AI Box. You are a helpful assistant."},
    {"role": "user", "content": "What is Renmin University of China?"}
]
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)

# Completion
output = model.generate(inputs["input_ids"], max_new_tokens=100, temperature=0.5)
print(tokenizer.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True))
```

**vLLM Serve Example**
```bash
vllm serve yulan-team/YuLan-Mini-Instruct --dtype bfloat16
```

**SGLang Serve Example**
```bash
python -m sglang.launch_server --model-path yulan-team/YuLan-Mini-Instruct --port 30000 --host 0.0.0.0
```

**Ollama**
```bash
ollama run hf.co/mradermacher/YuLan-Mini-Instruct-GGUF:IQ4_XS
```

---

## Contributing

We welcome any form of contribution, including feedback on model bad cases, feature suggestions, and example contributions. You can do so by submitting an [issue](https://github.com/RUC-GSAI/YuLan-Mini/issues).

## The Team

YuLan-Mini is developed and maintained by [AI Box, Renmin University of China](http://aibox.ruc.edu.cn/).

## License

- The code in this repository, the model weights, and optimizer states are released under the [MIT License](./LICENSE).
- Policies regarding the use of model weights, intermediate optimizer states, and training data will be announced in future updates.
- Limitations: Despite our efforts to mitigate safety concerns and encourage the generation of ethical and lawful text, the probabilistic nature of language models may still lead to unexpected outputs. For instance, responses might contain bias, discrimination, or other harmful content. Please refrain from disseminating such content. We are not liable for any consequences arising from the spread of harmful information.

## Citation

If you find YuLan-Mini helpful for your research or development, please cite [our technical report](https://arxiv.org/abs/2412.17743) and blog:


```
@article{hu2024yulan,
  title={YuLan-Mini: An Open Data-efficient Language Model},
  author={Hu, Yiwen and Song, Huatong and Deng, Jia and Wang, Jiapeng and Chen, Jie and Zhou, Kun and Zhu, Yutao and Jiang, Jinhao and Dong, Zican and Zhao, Wayne Xin and others},
  journal={arXiv preprint arXiv:2412.17743},
  year={2024}
}

@article{YuLan-Mini-Instruct,
  title={YuLan-Mini-Instruct Technical Report},
  author={RUCAIBox YuLan-Mini-Instruct Team},
  url={https://github.com/RUC-GSAI/YuLan-Mini},
  year={2025}
}
```


