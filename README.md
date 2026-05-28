# culturePRM

This repository contains scripts for processing CultureBank/NormAd-style cultural norm data, generating yes/no scenarios, generating reasoning traces, and evaluating Best-of-N outputs with PRM/ORM reward models.

## Directory

```text
culturePRM/
├── data/
│   ├── prm_train_labeled.jsonl
│   └── scenarios_filtered.jsonl
├── pipeline/
│   ├── step1_filter.py
│   ├── step2_score_norm.py
│   ├── step3_generate_scenarios.py
│   └── step4_generate_traces.py
└── eval/
    ├── generate_traces_rao_prompt.py
    └── eval_with_cached_traces_rao_fast.py
```

## Requirements

Install the main Python dependencies:

```bash
pip install torch transformers datasets vllm tqdm setproctitle
```

Some scripts also depend on project-local modules such as `utils.py`, `generation.py`, `prompt.py`, `train_prm.py`, and `train_orm.py`. Run commands from the repository root unless noted otherwise.

## Pipeline Scripts

### 1. Filter CultureBank Data

`pipeline/step1_filter.py` filters CultureBank JSONL data using simple quality criteria.

Filtering criteria:

- `agreement >= min_agreement`
- `actor_behavior` has at least 8 words
- vague or overly abstract behavior descriptions are removed

```bash
python pipeline/step1_filter.py \
  --input data/culturebank_reddit.jsonl \
  --output data/filtered.jsonl \
  --min_agreement 0.7
```

Key arguments:

- `--input`: input JSONL file
- `--output`: output JSONL file
- `--min_agreement`: minimum agreement threshold
- `--limit`: optional number of rows to process

### 2. Score Norm-Likeness

`pipeline/step2_score_norm.py` scores whether each filtered item expresses a cultural norm using a GPT-OSS-style model.

The output JSONL contains `source_id`, `is_norm`, `norm_strength`, `category`, `reason`, and `raw_output`.

```bash
python pipeline/step2_score_norm.py \
  --input data/filtered.jsonl \
  --output data/norm_scores/norm_gpt_oss_20b.jsonl \
  --model_path /path/to/gpt-oss-20B \
  --reasoning_effort high \
  --batch_size 16 \
  --tensor_parallel 2
```

Key arguments:

- `--input`: filtered JSONL file
- `--output`: norm scoring output JSONL
- `--model_path`: model path to load with vLLM
- `--reasoning_effort`: one of `low`, `medium`, `high`
- `--batch_size`: batch size
- `--tensor_parallel`: number of tensor parallel GPUs
- `--max_model_len`: model context length
- `--gpu_memory_utilization`: GPU memory utilization ratio
- `--max_tokens`: maximum generation tokens
- `--limit`: optional number of rows to process

### 3. Generate Yes/No Scenarios

`pipeline/step3_generate_scenarios.py` generates two short stories for each norm item:

- `yes`: a scenario where the character follows the norm
- `no`: a scenario where the character violates the norm

```bash
python pipeline/step3_generate_scenarios.py \
  --input data/split_a_prm.jsonl \
  --output data/scenarios_a.jsonl \
  --model_id /path/to/model \
  --max_model_len 4096
```

Key arguments:

- `--input`: input JSONL file for scenario generation
- `--output`: output JSONL file for generated scenarios
- `--model_id`: model path used by Transformers/vLLM
- `--max_model_len`: model context length
- `--limit`: optional number of rows to process

### 4. Generate Reasoning Traces

`pipeline/step4_generate_traces.py` generates reasoning traces and final answers for generated scenarios.

Supported prompt types:

- `A`: country + story
- `B`: country + rule-of-thumb + story
- `C`: country + background + story

```bash
python pipeline/step4_generate_traces.py \
  --input data/scenarios_a.jsonl \
  --output data/traces_A.jsonl \
  --model_path /path/to/model \
  --prompt_type A \
  --reasoning_effort high \
  --batch_size 16 \
  --tensor_parallel 2
```

Key arguments:

- `--input`: scenario JSONL file
- `--output`: trace output JSONL file
- `--model_path`: model path for reasoning trace generation
- `--prompt_type`: one of `A`, `B`, `C`
- `--reasoning_effort`: one of `low`, `medium`, `high`
- `--batch_size`: batch size
- `--tensor_parallel`: number of tensor parallel GPUs
- `--max_model_len`: model context length
- `--max_tokens`: maximum generation tokens
- `--gold_label_filter`: process only `yes`, `no`, or `neutral` examples
- `--model_type`: `gptoss` or `default`
- `--start_source_id`: resume from a specific `source_id`
- `--limit`: optional number of rows to process

## Evaluation Scripts

### Generate Rao Prompt Traces

`eval/generate_traces_rao_prompt.py` applies Rao et al.-style prompts to the NormAd dataset and stores N reasoning completions per example in a cache JSONL file.

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval/generate_traces_rao_prompt.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --conditioning rot \
  --n_samples 8 \
  --temperature 0.7 \
  --top_p 1.0 \
  --tensor_parallel 2 \
  --output cache/rao_traces_n8_seed42_rot.jsonl
```

Supported conditioning modes:

- `rot`: rule-of-thumb only
- `country`: country only
- `cval`: country + cultural value

Key arguments:

- `--model_path`: model path for completion generation
- `--conditioning`: one of `rot`, `country`, `cval`
- `--output`: trace cache JSONL path
- `--n_samples`: number of completions per prompt
- `--temperature`, `--top_p`, `--top_k`: sampling parameters
- `--max_new_tokens`: maximum new tokens
- `--tensor_parallel`: number of tensor parallel GPUs
- `--seed`: random seed
- `--gpu_memory_utilization`: GPU memory utilization ratio
- `--max_model_len`: model context length
- `--max_examples`: optional number of examples to process

### Evaluate Cached Traces with PRM/ORM

`eval/eval_with_cached_traces_rao_fast.py` scores cached reasoning traces with a PRM or ORM reward model and computes Best-of-N metrics.

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval/eval_with_cached_traces_rao_fast.py \
  --traces_cache cache/rao_traces_n8_seed42_rot.jsonl \
  --reward_model_path /path/to/reward_model \
  --reward_type prm \
  --prm_aggregation last \
  --batch_size 1024 \
  --output results/rao_bon_prm_last.jsonl
```

Outputs:

- `--output`: JSONL file with per-example predictions and scores
- `*.metrics.json`: summary metrics such as accuracy, binary metrics, and label distributions

Key arguments:

- `--traces_cache`: cache JSONL produced by `generate_traces_rao_prompt.py`
- `--reward_model_path`: PRM/ORM model path
- `--output`: evaluation result JSONL path
- `--reward_type`: `prm` or `orm`
- `--prm_aggregation`: one of `min`, `mean`, `last`
- `--seed`: random baseline seed
- `--batch_size`: batch size per GPU
- `--n_use`: number of cached completions to use
- `--skip_on_error`: create empty metric output and exit when the cache is invalid

## Example Workflow

CultureBank-based data generation:

```bash
python pipeline/step1_filter.py \
  --input data/culturebank_reddit.jsonl \
  --output data/filtered.jsonl

python pipeline/step2_score_norm.py \
  --input data/filtered.jsonl \
  --output data/norm_scores/norm_gpt_oss_20b.jsonl \
  --model_path /path/to/gpt-oss-20B

python pipeline/step3_generate_scenarios.py \
  --input data/split_a_prm.jsonl \
  --output data/scenarios_a.jsonl \
  --model_id /path/to/model

python pipeline/step4_generate_traces.py \
  --input data/scenarios_a.jsonl \
  --output data/traces_A.jsonl \
  --model_path /path/to/model \
  --prompt_type A
```

NormAd/Rao Best-of-N evaluation:

```bash
python eval/generate_traces_rao_prompt.py \
  --model_path /path/to/base_model \
  --conditioning rot \
  --n_samples 8 \
  --output cache/rao_traces_n8_seed42_rot.jsonl

python eval/eval_with_cached_traces_rao_fast.py \
  --traces_cache cache/rao_traces_n8_seed42_rot.jsonl \
  --reward_model_path /path/to/reward_model \
  --reward_type prm \
  --prm_aggregation last \
  --output results/rao_bon_prm_last.jsonl
```

## Output Formats

### Scenario JSONL

Example output from `step3_generate_scenarios.py`:

```json
{
  "id": "source_id_yes",
  "source_id": "source_id",
  "cultural_group": "...",
  "context": "...",
  "actor": "...",
  "actor_behavior": "...",
  "gold_label": "yes",
  "story": "...",
  "generation_model": "/path/to/model"
}
```

### Trace JSONL

Example output from `step4_generate_traces.py`:

```json
{
  "id": "...",
  "source_id": "...",
  "gold_label": "no",
  "story": "...",
  "prompt_type": "A",
  "reasoning_effort": "high",
  "thinking": "...",
  "final_answer": "yes",
  "raw_output": "..."
}
```

### Rao Cache JSONL

Example output from `generate_traces_rao_prompt.py`:

```json
{
  "id": "...",
  "country": "...",
  "story": "...",
  "rule_of_thumb": "...",
  "value": "...",
  "gold": "yes",
  "completions": ["...", "..."],
  "prompt_used": "rao",
  "conditioning": "rot"
}
```

## Notes

- Run scripts from the repository root: `/Users/jieun/Documents/culturePRM`.
- vLLM-based scripts assume an available GPU environment.
- `step2_score_norm.py` and `step4_generate_traces.py` support resume behavior when the output file already exists.
- PRM evaluation splits reasoning traces into sentence-level steps and aggregates completion scores with `min`, `mean`, or `last`.
