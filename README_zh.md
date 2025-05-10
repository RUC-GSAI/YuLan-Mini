<h4 align="center">
    <p>
        <b>中文</b> | <a href="https://github.com/RUC-GSAI/YuLan-Mini">English</a> | <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/README_ja.md">日本語</a>
    <p>
</h4>

<div align=center>
<img src="assets/YuLan-logo.png" width="400px">
<h1>YuLan-Mini: 数据高效的开源语言模型</h1>
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="license"></a>
<a href="https://arxiv.org/abs/2412.17743" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3"><img alt="Static Badge" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a><img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-Mini"></a>
</div>

YuLan-Mini 是一个 2.4B 参数量的轻量化语言模型。仅使用 1.08T Tokens 进行预训练，却达到了与使用更多数据的行业领先模型相媲美的性能，尤其是 **数学** 和 **代码** 两个领域。为方便复现，我们将开源相关预训练资源。

---

## 新闻

- [2025.01.29] YuLan-Mini-Instruct-v1 发布
- [2024.12.23] YuLan-Mini 及预训练资源发布

## 模型下载 🔗

> YuLan-Mini 是 [YuLan 系列](https://github.com/RUC-GSAI/YuLan-Chat) 的一部分，该系列还包括更大规模和不同训练策略的模型。

|  模型  | 上下文长度 | SFT | 🤗 Hugging Face | ModelScope | Wise Model |
|---------|----------------|-----|-----------------|------------|------------|
| YuLan-Mini | 28K | ❎ | [`Base`](https://huggingface.co/yulan-team/YuLan-Mini) | [`Base`](https://modelscope.cn/models/yulan-team/YuLan-Mini) | [`Base`](https://wisemodel.cn/models/yulan-team/YuLan-Mini) |
| YuLan-Mini-Instruct | 28K | ✅ | [`Instruct`](https://huggingface.co/yulan-team/YuLan-Mini-Instruct) | | |

> 中间检查点可以在[这里](#%E9%A2%84%E8%AE%AD%E7%BB%83%E8%B5%84%E6%BA%90-)找到。

---

## 能力介绍 🌟

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/main.png">
</div>

我们的预训练方法通过以下三项关键技术改进提升了训练效率：

1. 精细的数据处理流程，将数据清洗与数据课程策略相结合；
2. 稳定的优化方法，有效缓解预训练中的不稳定性；
3. 高效的退火策略，融合了目标数据选择和长上下文训练。

最终，使用我们的高效预训练策略，仅 1T 的数据量便可在数学和代码等领域，媲美 Qwen2.5-1.5B 在 18T 数据上的效果。我们将开源使用到的 1T 数据，其中指令数据仅占 3.5%。

---
## 基准测试 🌟

| Models                  | MMLU | CEVAL | GSM8K | ARC_CHALLENGE | GPQA | MATH | HUMANEVAL@1 | MBPP@1 |
|-------------------------|-------|-------|-------|---------------|------|------|-------------|--------|
| Qwen-2.5-1.5B-Instruct  | 57.5  | 65.4  | 73.2  | 47.8          | 29.8 | 55.2 | 61.6        | 88.1   |
| Llama3.2-3B-Instruct    | 60    | 45.9  | 43.4  | 78.6          | 38.6 | 48   | 51.5        | 80.4   |
| YuLan-Mini-Instruct  | 53.6  | 50.5    | 82.3  | 51.8          | 30.1 | 55.2 | 67.7        | 85.7   |

> 注意：模型大小的计算包含了嵌入层（embedding）的大小。

|      Models      | Model Size | # Train Tokens | Context Length | MATH 500 | GSM 8K | Human Eval | MBPP   | RACE Middle | RACE High | RULER  |
|:----------------|----------:|--------------:|--------------:|:--------|:------|:----------|:------|:-----------|:---------|:------|
|     MiniCPM      |    2.6B    |     1.06T      |       4K       |   15.00  |  53.83 |     50.00* |  47.31 |     56.61   |   44.27   |   N/A  |
|      Qwen-2      |    1.5B    |       7T       |      128K      |   22.60  | 46.90* |     34.80* | 46.90* |     55.77   |   43.69   |  60.16 |
|     Qwen2.5      |    0.5B    |      18T       |      128K      |   23.60  | 41.60* |     30.50* | 39.30* |     52.36   |   40.31   |  49.23 |
|     Qwen2.5      |    1.5B    |      18T       |      128K      |   **45.40**  | **68.50\*** |     37.20* | 60.20* |     **58.77**   |   44.33   |  <ins>68.26</ins> |
|     Gemma2       |    2.6B    |       2T       |       8K       |   18.30* | 30.30* |     19.50* | 42.10* |       -     |      -    |   N/A  |
|    StableLM2     |    1.7B    |       2T       |       4K       |     -    |  20.62 |      8.50* |  17.50 |     56.33   |   **45.06**   |   N/A  |
|    SmolLM2       |    1.7B    |      11T       |       8K       |   11.80  |    -   |     23.35  |  45.00 |     55.77   |   43.06   |   N/A  |
|    Llama3.2      |    3.2B    |       9T       |      128K      |    7.40  |    -   |     29.30  |  49.70 |     55.29   |   43.34   |  **77.06** |
|    YuLan-Mini    |    2.4B    |     1.04T      |       4K       |   32.60  |  66.65 |     <ins>61.60</ins>  |  **66.70** |     55.71   |   43.58   |   N/A  |
|    YuLan-Mini    |    2.4B    |     1.08T      |      28K       |  <ins>37.80</ins>  |  <ins>68.46</ins> |    **64.00**  |  <ins>65.90</ins>|     <ins>57.18</ins>   |   <ins>44.57</ins>   |  51.48 |


|      Models      | LAMBADA | MMLU  | CMMLU | CEval | HellaSwag | WinoGrande | StoryCloze | ARC-e | ARC-c |
|:----------------|:-------|:-----|:-----|:-----|:----------|:-----------|:-----------|:-----|:-----|
|   MiniCPM-2.6B   |  61.91  | 53.37 | 48.97 | 48.24 |   67.92    |     65.74   |     78.51   | 55.51 | 43.86 |
|   Qwen2-1.5B     |  64.68  | 55.90 | **70.76** | **71.94** |   66.11    |     66.14   |     77.60   | 62.21 | 42.92 |
|  Qwen2.5-0.5B    |  52.00  | 47.50 | 52.17 | 54.27 |   50.54    |     55.88   |     71.67   | 56.10 | 39.51 |
|  Qwen2.5-1.5B    |  62.12  | <ins>60.71</ins> | <ins>67.82</ins> | <ins>69.05</ins> |   67.18    |     64.48   |     76.80   | **71.51** | <ins>53.41</ins> |
|   Gemma2-2.6B    |    -    | 52.20*|   -   | 28.00*|   <ins>74.60*</ins>   |    **71.50\***   |       -     |   -   | **55.70\***|
| StableLM2-1.7B   |  66.15  | 40.37 | 29.29 | 26.99 |   69.79    |     64.64   |     <ins>78.56</ins>   | 54.00 | 40.78 |
|  SmolLM2-1.7B    |  <ins>67.42</ins>  | 51.91 | 33.46 | 35.10 |   72.96    |     67.40   |     **79.32**   | 44.82 | 35.49 |
|   Llama3.2-3B    |  **69.08**  | **63.40** | 44.44 | 44.49 |   **75.62**    |     <ins>67.48</ins>   |     76.80   | <ins>70.12</ins> | 48.81 |
|    YuLan-Mini    |  64.72  | 51.79 | 48.35 | 51.47 |   68.65    |     67.09   |     76.37   | 69.87 | 50.51 |
|    YuLan-Mini    |  65.67  | 49.10 | 45.45 | 48.23 |   67.22    |     67.24   |     75.89   | 67.47 | 49.32 |

---

## 预训练资源 🔧

为了提高研究的透明度和可复现性，我们开源了相关的[预训练资源](https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain)：

### 预训练

<details><summary>1. 预训练和评估代码</summary>

预训练代码可以在[这里](https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain)找到。请注意，由于后续的代码修改，此代码可能无法直接运行，可能需要进行一些调整。

<h4 id="step-1-modify-the-config-json-">步骤 1：修改 <code>config.json</code></h4>
<p>由于 Hugging Face Trainer 的实现，某些参数存储在 <code>config.json</code> 文件中，无法通过 Trainer 的命令行参数进行修改。因此，您需要首先更新 <code>config.json</code> 文件中的这些参数，特别是：</p>
<ul>
<li><strong><code>save_steps</code></strong>：保存中间检查点的频率。</li>
<li><strong><code>train_batch_size</code></strong>：每个 GPU 的批大小（相当于 Trainer 中的 <code>per_device_train_batch_size</code>）。在稳定训练阶段，我们使用了 1008 的批大小（大约 4M 个 token）。保持相同的批大小对于训练效果同样重要。</li>
</ul>
<p>以下是一个正确配置的 <code>config.json</code> 文件示例：</p>
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
<h4 id="step-2-enable-universal-checkpointing-in-the-deepspeed-configuration">步骤 2：在 DeepSpeed 配置中启用通用检查点</h4>
<p>为了确保 DeepSpeed 集成加载通用检查点，您需要在 DeepSpeed 配置 JSON 文件中启用此功能。</p>
<p>以下是一个启用了通用检查点的 ZeRO2 配置示例：</p>
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
<h4 id="step-3-resume-training">步骤 3：恢复训练</h4>
<p>调用 <code>trainer.train</code> 时，包含 <code>resume_from_checkpoint</code> 参数以从通用检查点加载分布式优化器状态并恢复训练。</p>
<pre><code class="lang-python"><span class="hljs-attr">trainer.train(resume_from_checkpoint</span>=<span class="hljs-string">training_args.resume_from_checkpoint)</span>
</code></pre>
<p>我们提供了一个内部<a href="https://github.com/RUC-GSAI/YuLan-Mini/tree/main/pretrain">训练框架</a>供您参考，但您可以自由选择其他框架。</p>

</details>

<details><summary>2. 中间阶段检查点</summary>
中间阶段检查点发布在 <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">YuLan-Mini</a> 中。

<table>
    <thead>
        <tr>
            <th>阶段</th>
            <th>课程阶段</th>
            <th>4K 上下文</th>
            <th>28K 上下文</th>
            <th>优化器</th>
            <th>推理架构</th>
            <th>LAMBADA <code>Acc</code></th>
            <th>GSM8K <code>Acc</code></th>
            <th>HumanEval <code>pass@1</code></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>稳定</td>
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
            <td>稳定</td>
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
            <td>稳定</td>
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
            <td>稳定</td>
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
            <td>稳定</td>
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
            <td>退火</td>
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
            <td>退火</td>
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

\*：为了更容易推理和部署，我们将重新参数化的附加参数和缩放因子合并到最终发布的模型中 ([**YuLan-Mini**](https://huggingface.co/yulan-team/YuLan-Mini) 和 **YuLan-Mini-Intermediate-4K**)，使其能够在 Llama 架构上运行。但是，这些参数仍然保留在训练过程的中间检查点中。

</details>

<details><summary>3. 退火前的优化器状态</summary>

<a href="https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing">🤗 YuLan-Mini-Before-Annealing</a>
</details>

### 数据集


<details><summary>4. 使用的开源数据集</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets">使用的开源数据集列表</a>

</details>

<details><summary>5. 每个阶段的数据分布</summary>

⬇️ 点击查看更多详情：
<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/datasets/final.pdf">
  <div align=center>
    <img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data_distribution_for_every_phase.png">
  </div>
</a>

</details>

<details><summary>6. 合成数据</summary>

<a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/preprocess">数据清洗</a> 和 <a href="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/pretrain/synthesis">合成</a> 流程：

<div align=center>
<img src="https://github.com/RUC-GSAI/YuLan-Mini/blob/main/assets/data-pipeline.png">
</div>

我们使用的合成数据发布在 <a href="https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3">🤗 YuLan-Mini-Datasets</a>

</details>


### 您可以使用这些预训练资源做什么

1. **预训练**您自己的 LLM。您可以使用[我们的数据](https://huggingface.co/yulan-team/YuLan-Mini-Datasets)和课程来训练一个与 YuLan-Mini 一样强大的模型。
2. 执行您自己的**学习率退火**。在退火阶段，YuLan-Mini 的学习能力达到顶峰。您可以从[退火前的检查点](https://huggingface.co/yulan-team/YuLan-Mini-Before-Annealing)恢复训练，并使用您自己的数据集进行学习率退火。
3. **微调** LLM 的 Instruct 版本。您可以使用 [YuLan-Mini](https://huggingface.co/yulan-team/YuLan-Mini) 基础模型来训练您自己的 Instruct 版本。
4. **训练动态**研究。您可以使用 YuLan-Mini 的[中间检查点](https://huggingface.co/collections/yulan-team/yulan-mini-676d214b24376739b00d95f3)来探索预训练过程中的内部变化。
5. **合成**您自己的数据。您可以使用 YuLan-Mini 的[数据流程](https://github.com/RUC-GSAI/YuLan-Mini)来清理和生成您自己的数据集。
---

## 快速开始 💻

以下是使用 Huggingface 的简单推理代码示例：

**Huggingface 推理示例**
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

**vLLM部署示例**
```bash
vllm serve yulan-team/YuLan-Mini-Instruct --dtype bfloat16
```

**SGLang部署示例**
```bash
python -m sglang.launch_server --model-path yulan-team/YuLan-Mini-Instruct --port 30000 --host 0.0.0.0
```

**Ollama部署示例**
```bash
ollama run hf.co/mradermacher/YuLan-Mini-Instruct-GGUF:IQ4_XS
```

---

## 贡献

我们欢迎任何形式的贡献，包括模型错误案例的反馈、功能建议和示例贡献。您可以通过提交[issue](https://github.com/RUC-GSAI/YuLan-Mini/issues)来贡献。

## 许可协议

- 本仓库代码使用 [MIT License](./LICENSE)。
- 局限性：尽管我们尝试减少模型在使用中可能出现的安全性问题，并鼓励模型生成符合道德和法律要求的文本，但由于语言模型基于概率生成的范式，模型仍然可能会产生意外的输出。例如，生成的响应可能包含偏见、歧视或其他有害内容。请不要传播此类内容。我们对因传播有害信息而造成的任何后果不承担任何责任。

## 引用

如果您发现 YuLan-Mini 对您的研究或开发有帮助，请引用我们的[技术报告](https://arxiv.org/abs/2412.17743)：

```BibTex
@article{hu2024yulan,
  title={YuLan-Mini: An Open Data-efficient Language Model},
  author={Hu, Yiwen and Song, Huatong and Deng, Jia and Wang, Jiapeng and Chen, Jie and Zhou, Kun and Zhu, Yutao and Jiang, Jinhao and Dong, Zican and Zhao, Wayne Xin and others},
  journal={arXiv preprint arXiv:2412.17743},
  year={2024}
}
```
