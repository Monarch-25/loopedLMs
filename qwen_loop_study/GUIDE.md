# Qwen Loop Study — Step-by-Step Guide

> **Research question**: After converting pretrained Qwen into a looped/shared-block architecture, does it match or beat dense Qwen on math reasoning after the same recovery, SFT, and GRPO budget?

This guide walks you through the full experiment pipeline, from environment setup to final evaluation and results analysis.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Weights & Biases Setup](#3-weights--biases-setup)
4. [Step 0 — Smoke Test](#step-0--smoke-test)
5. [Step 1 — Build Data Splits](#step-1--build-data-splits)
6. [Step 2 — Recovery Training](#step-2--recovery-training)
7. [Step 3 — Math SFT](#step-3--math-sft)
8. [Step 4 — GRPO (RL)](#step-4--grpo-rl)
9. [Step 5 — Evaluation](#step-5--evaluation)
10. [Step 6 — Results & Analysis](#step-6--results--analysis)
11. [Config Reference](#config-reference)
12. [Weights & Biases Dashboard Guide](#weights--biases-dashboard-guide)
13. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### Hardware

| Environment | Purpose | Minimum |
|---|---|---|
| **Local (MPS/CPU)** | Smoke tests, unit tests, config validation | Apple M-series or any modern CPU, 16 GB RAM |
| **GPU** | Real training and evaluation runs | 1× A100-40GB (0.5B models) or 1× A100-80GB (1.5B models) |

### Software

- **Python**: 3.11 or later
- **CUDA**: 12.1+ (GPU runs only)
- **Git**: for cloning the repo
- **wandb** (optional): for experiment tracking

---

## 2. Installation

### Clone the repository

```bash
git clone <repo-url> loopedLMs
cd loopedLMs/qwen_loop_study
```

### Install the package

There are two environment strategies, controlled by lockfiles:

#### Option A — Local smoke testing (MPS/CPU)

```bash
pip install -e ".[dev]"
```

This installs core dependencies only. Sufficient for unit tests and tiny-model correctness checks.

#### Option B — Full GPU environment (recommended for real runs)

```bash
pip install -e ".[eval,dev,tracking]"
```

This includes `math-verify`, `lighteval`, and `wandb` for complete experiment tracking.

### Verify installation

```bash
python -c "from qwen_loop_study.models import LoopedQwenForCausalLM; print('OK')"
```

---

## 3. Weights & Biases Setup

W&B integration is built into every stage of the pipeline. It is **optional** — all stages work without it — but strongly recommended for experiment comparison.

### Login

```bash
pip install wandb
wandb login
```

### Environment variables (alternative to YAML config)

```bash
export WANDB_PROJECT="qwen-loop-study"
export WANDB_ENTITY="your-team-name"  # optional
```

### Enable via YAML config

Every config file has a `wandb:` block. Set `enabled: true` to activate:

```yaml
training:
  report_to: ["wandb"]
  wandb:
    enabled: true
    project: qwen-loop-study
    watch: true           # log gradient histograms
    watch_log: gradients
    watch_log_freq: 100
```

### What gets logged

When W&B is enabled, the following are automatically tracked per stage:

| Stage | Logged Metrics |
|---|---|
| **Recovery** | CE loss, KL loss, combined loss, layer norms, compression ratio |
| **SFT** | Train/eval loss curves, dataset stats, model diagnostics |
| **GRPO** | Reward curves, KL penalty, completion lengths, loss |
| **Eval** | Accuracy (+ CI), parse rate, tokens/sec, per-example predictions table |

---

## Step 0 — Smoke Test

Before running any real experiments, verify the codebase works:

```bash
cd qwen_loop_study
PYTHONPATH=src pytest tests/ -v
```

Expected: all unit and integration tests pass. The tests use tiny synthetic models so they run on CPU in seconds.

### What the tests verify

- Layer-tying mapping is exact
- Looped forward output shape matches dense output shape
- Fixed-depth unroll is deterministic under a fixed seed
- `math-verify` reward extraction works on known examples
- End-to-end mini workflows (recovery → SFT → GRPO → eval) execute on tiny models

---

## Step 1 — Build Data Splits

The first real step is to create deterministic, decontaminated dataset splits for both SFT and GRPO stages.

### Build SFT splits

```bash
PYTHONPATH=src qwen-loop-build-splits \
    --config configs/sft/sft_dense_0p5b.yaml \
    --stage sft
```

This produces `data/prepared/sft/` with:
- **train**: ~25,000 problems, stratified by source and solution length
- **validation**: ~2,000 problems
- Decontaminated against MATH-500 and GSM8K (8-gram overlap removal)

### Build GRPO splits

```bash
PYTHONPATH=src qwen-loop-build-splits \
    --config configs/grpo/grpo_dense_0p5b.yaml \
    --stage grpo
```

This produces `data/prepared/grpo/` with:
- **train**: ~10,000 medium/hard problems
- **validation**: ~1,000 problems

> **Note**: You only need to build splits once — all dense and looped configs for the same size share the same prepared data directory.

---

## Step 2 — Recovery Training

Recovery is **mandatory** because we are post-hoc loopifying pretrained Qwen, not training from scratch. It stabilizes the looped model via knowledge distillation from the frozen dense teacher.

### What happens during recovery

1. The dense Qwen checkpoint is loaded as a frozen **teacher**.
2. A **student** (dense or looped) is initialized — for looped models, layers are averaged from groups of teacher layers.
3. The student trains on a mixed corpus (10M tokens from FineWeb-Edu + 2M tokens from OpenR1-Math).
4. Loss = `CE_weight × next-token CE` + `KL_weight × temperature² × KL(student ∥ teacher)`.

### Run all 4 conditions

```bash
# Dense 0.5B recovery
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_dense_0p5b.yaml

# Looped 0.5B recovery (24 layers → 6 unique × 4 steps)
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_0p5b.yaml

# Dense 1.5B recovery
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_dense_1p5b.yaml

# Looped 1.5B recovery (28 layers → 7 unique × 4 steps)
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_1p5b.yaml
```

### Output structure

```
runs/recovery/
├── dense-0p5b/
│   ├── model.safetensors
│   ├── config.json
│   ├── train_metrics.json
│   ├── eval_metrics.json
│   └── manifest.jsonl
├── looped-0p5b/
│   └── ...
├── dense-1p5b/
│   └── ...
└── looped-1p5b/
    └── ...
```

### Key metrics to watch

| Metric | Healthy range | Red flag |
|---|---|---|
| `train_loss` | Steadily decreasing | Plateaus above 3.0 |
| `eval_loss` | Tracks train_loss closely | Diverges (overfitting) |
| `layer_param_norm_std` | Low, stable | Spiking (layer collapse) |
| `parameter_compression_ratio` | ~4.0 for looped | Any other value |

### Resume from checkpoint

If a run is interrupted:

```bash
# Edit the config YAML:
# training:
#   resume_from_checkpoint: runs/recovery/looped-0p5b/checkpoint-500
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_0p5b.yaml
```

---

## Step 3 — Math SFT

Supervised fine-tuning on math problems teaches the model the `\boxed{}` output format and basic reasoning patterns.

### Prerequisites

- Recovery checkpoints from Step 2
- Prepared SFT splits from Step 1

### Run all 4 conditions

```bash
# Dense 0.5B SFT
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_dense_0p5b.yaml

# Looped 0.5B SFT
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_looped_0p5b.yaml

# Dense 1.5B SFT
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_dense_1p5b.yaml

# Looped 1.5B SFT
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_looped_1p5b.yaml
```

### What the SFT stage does

1. Loads the recovered checkpoint (dense or looped).
2. Formats each problem as a chat template: `system → user (problem) → assistant (solution with \boxed{})`.
3. Trains with standard causal LM loss via TRL's `SFTTrainer`.
4. Saves the fine-tuned model to `runs/sft/<condition>/`.

### Key metrics to watch

| Metric | What to look for |
|---|---|
| `train/loss` | Should reach < 1.0 within first epoch |
| `eval/loss` | Should track train loss — gap > 0.3 suggests overfitting |
| `model/virtual_depth` | Confirms looped model uses correct unroll depth (24 or 28) |

---

## Step 4 — GRPO (RL)

Group Relative Policy Optimization with verifiable math rewards. This is the final training stage.

### Prerequisites

- SFT checkpoints from Step 3
- Prepared GRPO splits from Step 1

### Run all 4 conditions

```bash
# Dense 0.5B GRPO
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_dense_0p5b.yaml

# Looped 0.5B GRPO
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_looped_0p5b.yaml

# Dense 1.5B GRPO
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_dense_1p5b.yaml

# Looped 1.5B GRPO
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_looped_1p5b.yaml
```

### How GRPO works in this study

1. The model generates `num_generations=4` completions per prompt.
2. Each completion is scored by the reward function:
   - **1.0** — exact verified match via `math-verify`
   - **0.05** — valid `\boxed{}` output but wrong answer
   - **0.0** — no parseable answer
3. The DAPO-style loss optimizes the policy using relative advantages within each group.
4. KL penalty (`beta=0.01`) prevents the model from drifting too far from the SFT policy.

### Key metrics to watch

| Metric | What to look for |
|---|---|
| `train/loss` | Should decrease and stabilize |
| Mean reward | Should increase from ~0.0-0.1 toward 0.3+ |
| Reward std | Very high variance = unstable training |
| Completion length | Sudden drops may indicate mode collapse |

---

## Step 5 — Evaluation

Run the final models on the primary benchmarks.

### Run all 4 conditions

```bash
# Dense 0.5B eval
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_dense_0p5b.yaml

# Looped 0.5B eval
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_looped_0p5b.yaml

# Dense 1.5B eval
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_dense_1p5b.yaml

# Looped 1.5B eval
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_looped_1p5b.yaml
```

### What gets measured

| Benchmark | Metric | Details |
|---|---|---|
| **MATH-500** | Exact-match accuracy | 500-problem subset with 95% bootstrap CI |
| **GSM8K** | Exact-match accuracy | Grade-school math with `math-verify` scoring |

### Additional metrics collected

- **Parse rate**: fraction of outputs containing a valid `\boxed{}` answer
- **Average completion length**: word count of generated solutions
- **Average loop steps**: should be `virtual_depth` for looped models
- **Tokens/sec**: inference throughput
- **Wall-clock time**: total evaluation time

### Output files

```
results/
├── summary.csv                                  # One row per benchmark × condition
├── per_example_predictions_dense_0p5b.parquet   # Full predictions for analysis
├── per_example_predictions_looped_0p5b.parquet
├── per_example_predictions_dense_1p5b.parquet
└── per_example_predictions_looped_1p5b.parquet
```

---

## Step 6 — Results & Analysis

### Read the summary

```python
import pandas as pd

df = pd.read_csv("results/summary.csv")
print(df[["benchmark", "accuracy", "ci_low", "ci_high", "parse_rate", "tokens_per_second"]])
```

### Compare dense vs looped

```python
# Load per-example predictions
dense = pd.read_parquet("results/per_example_predictions_dense_0p5b.parquet")
looped = pd.read_parquet("results/per_example_predictions_looped_0p5b.parquet")

# Problems that looped gets right but dense doesn't
looped_wins = looped[looped["correct"] & ~dense["correct"]]
print(f"Looped-only correct: {len(looped_wins)}")
```

### Final report

The report template is at `reports/qwen_looplm_math_study.md`. It expects:
- Benchmark tables with CIs
- Compute tables (FLOPs, tokens/sec, peak memory)
- GRPO reward curves
- Accuracy vs compute plots

---

## Config Reference

### Model block

```yaml
model:
  model_name_or_path: Qwen/Qwen2.5-0.5B   # HF hub path or local checkpoint
  architecture: dense                        # "dense" or "looped"
  torch_dtype: bfloat16                      # "auto", "float32", "bfloat16", "float16"
  recurrent_steps: 4                         # R — loop unroll depth (looped only)
  num_unique_layers: 6                       # Physical layer count (looped only)
  enable_exit_head: false                    # Adaptive early exit (experimental)
```

### Training block

```yaml
training:
  output_dir: runs/sft/dense-0p5b
  seed: 42
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32            # Effective batch = 32
  learning_rate: 2.0e-5
  warmup_ratio: 0.03
  num_train_epochs: 2
  max_steps: -1                              # -1 = use num_train_epochs
  logging_steps: 10
  eval_steps: 100
  save_steps: 100
  save_total_limit: 2                        # Keep only last 2 checkpoints
  gradient_checkpointing: true               # Required for 1.5B models
  bf16: true
  report_to: ["wandb"]                       # [] to disable W&B
  resume_from_checkpoint: null               # Path to resume from
  wandb:
    enabled: true
    project: qwen-loop-study
    watch: true
    watch_log: gradients
    watch_log_freq: 100
```

### Recovery block

```yaml
recovery:
  teacher_model_name_or_path: Qwen/Qwen2.5-0.5B
  ce_weight: 1.0
  kl_weight: 1.0
  temperature: 1.0
  max_total_tokens: 12000000
  fineweb_token_budget: 10000000
  math_token_budget: 2000000
```

### GRPO block

```yaml
grpo:
  beta: 0.01                    # KL penalty coefficient
  num_generations: 4            # Completions per prompt
  max_prompt_length: 512
  max_completion_length: 256
  temperature: 0.9              # Sampling temperature for generation
  loss_type: dapo               # DAPO-style loss variant
  reward:
    exact_match_reward: 1.0
    boxed_parse_reward: 0.05
    incorrect_reward: 0.0
```

### W&B block

```yaml
wandb:
  enabled: true                  # Master toggle
  project: qwen-loop-study       # W&B project name
  entity: null                   # W&B team (optional)
  group: null                    # Group related runs
  name: null                     # Custom run name (auto-generated if null)
  tags: []                       # Additional tags
  mode: online                   # "online", "offline", or "disabled"
  save_code: true                # Snapshot source code
  watch: true                    # Log gradient/parameter histograms
  watch_log: gradients           # "gradients", "parameters", or "all"
  watch_log_freq: 100            # How often to log histograms
```

---

## Weights & Biases Dashboard Guide

When W&B is enabled, every run logs to your project. Here's how to use the dashboard effectively for this study.

### Recommended dashboard layout

1. **Group by architecture** (`dense` vs `looped`) to get side-by-side comparisons.
2. **Filter by stage** (`recovery`, `sft`, `grpo`, `eval`) using the `job_type` field.
3. **Create a comparison table** with columns: `model/architecture`, `eval/accuracy`, `eval/parse_rate`, `model/parameter_count`.

### Key charts to create

| Chart | X-axis | Y-axis | Purpose |
|---|---|---|---|
| Recovery convergence | `trainer/global_step` | `train/loss` | Verify looped recovery is stable |
| SFT learning curve | `trainer/global_step` | `eval/loss` | Compare convergence speed |
| GRPO reward curve | `trainer/global_step` | `grpo/mean_reward` | Track RL progress |
| Layer norm drift | `trainer/global_step` | `model/layer_*_param_norm` | Detect weight collapse in shared layers |
| Final accuracy | `benchmark` | `eval/accuracy` | The bottom-line comparison |

### Artifacts

Each stage saves its output checkpoint and metrics as W&B artifacts, enabling full lineage tracking from recovery → SFT → GRPO → eval.

---

## Troubleshooting

### OOM (Out of Memory)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix**: Reduce `per_device_train_batch_size` or enable `gradient_checkpointing: true` in the config. For GRPO, also reduce `num_generations` or `max_completion_length`.

### Tokenizer warnings

```
Special tokens have been added in the vocabulary...
```

**Safe to ignore.** The tokenizer automatically sets `pad_token = eos_token` when no pad token is defined.

### math-verify import errors

```
ImportError: No module named 'math_verify'
```

**Fix**: Install the eval extras:
```bash
pip install -e ".[eval]"
```

### W&B not logging

**Checklist:**
1. Is `wandb` installed? `pip install wandb`
2. Are you logged in? `wandb login`
3. Is `enabled: true` in the config's `wandb:` block?
4. Is `report_to` set to `["wandb"]` (not `[]`)?

### Looped model produces garbage

This usually means recovery was insufficient. Check:
- `recovery/kl_loss` should be decreasing toward < 1.0
- `model/layer_param_norm_std` should be small — large std means layers diverged
- Try running recovery for more steps (`max_steps: 4000`)

### Resume from checkpoint

Every stage supports resuming:

```yaml
training:
  resume_from_checkpoint: runs/sft/looped-0p5b/checkpoint-500
```

The trainer will pick up from the saved optimizer state, scheduler, and RNG.

---

## Full Pipeline — Quick Reference

```bash
# 0. Install
pip install -e ".[eval,dev,tracking]"

# 1. Build data splits (once)
PYTHONPATH=src qwen-loop-build-splits --config configs/sft/sft_dense_0p5b.yaml --stage sft
PYTHONPATH=src qwen-loop-build-splits --config configs/grpo/grpo_dense_0p5b.yaml --stage grpo

# 2. Recovery (4 runs)
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_dense_0p5b.yaml
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_dense_1p5b.yaml
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_1p5b.yaml

# 3. SFT (4 runs)
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_dense_0p5b.yaml
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_dense_1p5b.yaml
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_looped_1p5b.yaml

# 4. GRPO (4 runs)
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_dense_0p5b.yaml
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_dense_1p5b.yaml
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_looped_1p5b.yaml

# 5. Evaluation (4 runs)
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_dense_0p5b.yaml
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_dense_1p5b.yaml
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_looped_1p5b.yaml

# 6. Results
python -c "import pandas as pd; print(pd.read_csv('results/summary.csv').to_string())"
```
