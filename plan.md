# Qwen-Only Study: Post-Hoc Looped Qwen vs Dense Qwen on Math

## Summary
- Use one model family only: `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B`.
- Do not pretrain from scratch.
- Main question: after converting pretrained Qwen into a looped/shared-block architecture, does it match or beat dense Qwen on math after the same recovery, SFT, and GRPO budget?
- Main baseline: standard verifiable-reward RL using TRL `GRPOTrainer` with the DAPO-style loss variant.
- Primary evals: `HuggingFaceH4/MATH-500` and `openai/gsm8k`, scored with `math-verify` where applicable.
- Methodology follows `ml-intern`: literature first, validate datasets, use current HF examples/docs, run one pilot before any sweep, and attribute every result to a concrete recipe.

## Model Choice
- Required models:
  - [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B)
  - [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- Reason for `Qwen2.5` over `Qwen3`:
  - cleaner same-family scale pair
  - closer precedent in [`open-r1`](https://github.com/huggingface/open-r1)
  - easier comparison against existing GRPO recipes
- Dense conditions:
  - dense-0.5B
  - dense-1.5B
- Looped conditions:
  - looped-0.5B
  - looped-1.5B

## Architecture Conversion
- Keep tokenizer, embeddings, RoPE, LM head, and hidden size unchanged.
- Convert by tying decoder blocks and unrolling recurrently at fixed depth `R=4`.
- For `Qwen2.5-0.5B`:
  - 24 dense layers -> 6 unique blocks repeated 4 times.
  - Initialize shared block `j` from the mean of original layers `{j, j+6, j+12, j+18}`.
- For `Qwen2.5-1.5B`:
  - 28 dense layers -> 7 unique blocks repeated 4 times.
  - Initialize shared block `j` from the mean of original layers `{j, j+7, j+14, j+21}`.
- First pass uses fixed-depth looping only.
- Add adaptive early exit only if the fixed-depth looped model reaches within 3 absolute points of dense SFT on `MATH-500`.

## Training Stages
1. Environment setup
- Local smoke env: current `torch` conda env on M2 for tiny-model correctness checks only.
- GPU env: Linux CUDA env with current `transformers`, current `trl`, `accelerate`, `datasets`, `vllm`, `math-verify`, and `lighteval`.
- Keep one lockfile for local smoke and one for GPU runs.

2. Recovery stage after surgery
- This is mandatory because we are not doing from-scratch looped pretraining.
- Freeze the original dense checkpoint as the teacher.
- Train both dense and looped students for the same number of steps on the same recovery corpus.
- Recovery corpus:
  - `HuggingFaceFW/fineweb-edu` sampled to 10M tokens
  - plus 2M tokens of raw text from `open-r1/OpenR1-Math-220k`
- Loss:
  - dense student: next-token CE + KL to frozen dense teacher
  - looped student: next-token CE + KL to frozen dense teacher
- Goal: equalize adaptation budget and stabilize the looped model before math tuning.

3. Math SFT
- Dataset: [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- Build deterministic splits:
  - 25k train
  - 2k validation
- Stratify by source and solution length.
- Prompt format:
  - require reasoning
  - require final answer in `\boxed{}`
- Sequence lengths:
  - 1024 for 0.5B
  - 1536 for 1.5B
- Sweep once on a pilot subset:
  - learning rate `{1e-5, 2e-5, 5e-5}`
  - epochs `{1, 2}`
- Freeze one SFT recipe per size and reuse it for dense and looped.

4. RL stage
- Main RL dataset: [`open-r1/Big-Math-RL-Verified-Processed`](https://huggingface.co/datasets/open-r1/Big-Math-RL-Verified-Processed)
- Use only medium/hard bins in the first pass.
- Split:
  - 10k train
  - 1k validation
- RL method:
  - TRL `GRPOTrainer`
  - `loss_type="dapo"`
  - fixed-depth generation for looped models during training
  - no adaptive exit during RL
- Reward recipe:
  - `1.0` exact verified final-answer match
  - `0.05` valid boxed final answer with parse success
  - `0.0` otherwise
- RL pilot sweep:
  - learning rate `{5e-6, 1e-5}`
  - beta `{0.0, 0.01}`
  - max completion length `{256, 384}`
- Freeze one RL recipe per size.

## Evaluation
- Primary benchmarks:
  - [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
  - [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- Secondary benchmark if budget remains:
  - AIME24 via `lighteval`
- Primary metrics:
  - exact-match accuracy
  - 95% bootstrap confidence intervals
- Secondary metrics:
  - parse rate
  - average completion length
  - average loop steps
  - tokens/sec
  - peak memory
  - wall-clock time
  - estimated train FLOPs
  - accuracy per unit compute
- Use [`math-verify`](https://github.com/huggingface/Math-Verify) for answer extraction and verification.

## Experiment Matrix
- Base pretrained dense checkpoints:
  - dense-0.5B
  - dense-1.5B
- After recovery:
  - dense-recovered-0.5B
  - looped-recovered-0.5B
  - dense-recovered-1.5B
  - looped-recovered-1.5B
- After SFT:
  - dense-sft-0.5B
  - looped-sft-0.5B
  - dense-sft-1.5B
  - looped-sft-1.5B
- After GRPO:
  - dense-grpo-0.5B
  - looped-grpo-0.5B
  - dense-grpo-1.5B
  - looped-grpo-1.5B
- Optional gated inference only on promising looped models after GRPO.

## Local Setup and Validation
- Use the local `torch` conda env only for toy correctness checks on M2.
- Required local checks:
  - tiny Qwen-like model forward pass
  - layer-tying conversion correctness
  - save/load of looped checkpoints
  - one tiny SFT step
  - one tiny GRPO step
  - `math-verify` reward extraction
- Do not use local MPS runs for real throughput or final benchmark claims.

## Code Structure
- `src/models/looped_qwen.py`
  - shared-block Qwen wrapper
  - fixed-depth unroll
  - optional gate head
- `src/training/recover.py`
  - dense-teacher distillation recovery stage
- `src/training/sft.py`
  - shared dense/looped SFT entrypoint
- `src/training/grpo.py`
  - shared dense/looped GRPO entrypoint
- `src/data/build_splits.py`
  - deterministic split creation and decontamination
- `src/eval/run_eval.py`
  - unified benchmark runner
- `configs/`
  - one frozen YAML per condition
- `results/`
  - per-run metrics and per-example outputs
- `reports/`
  - final paper-style summary

## Test Cases
- Unit tests:
  - tied-layer mapping is exact
  - looped forward output shape matches dense output shape
  - fixed-depth unroll is deterministic under a fixed seed
  - gate probabilities sum to 1
  - `math-verify` reward matches known gold examples
- Integration tests:
  - recovery stage on a tiny model
  - SFT on a tiny split
  - GRPO on a tiny split
  - checkpoint resume
  - end-to-end eval file generation

## Acceptance Criteria
- Recovery stage completes for dense and looped students at both sizes.
- Looped SFT does not collapse and produces valid boxed outputs.
- All GRPO conditions finish at least one stable run.
- Final report answers all three:
  - does loopification recover after surgery?
  - does looped Qwen beat dense Qwen after SFT?
  - does looped Qwen benefit from GRPO as much as dense Qwen?

## Report Output
- `reports/qwen_looplm_math_study.md`
- Required sections:
  - exact research question
  - why this is not a strict Ouro replication
  - model conversion recipe
  - datasets and filtering
  - training recipes
  - benchmark tables
  - compute tables
  - failure analysis
  - conclusion
- Required plots:
  - accuracy vs compute
  - dense vs looped by stage
  - GRPO reward curves
  - loop-step histogram
  - parse-rate comparison
- Required artifacts:
  - `results/summary.csv`
  - `results/per_example_predictions.parquet`
  - frozen config files for every reported run

## Assumptions and Defaults
- Qwen-only is the final family choice.
- No from-scratch pretraining.
- The scientific claim is explicitly post-hoc loopification of pretrained Qwen, not faithful Ouro pretraining.
- `R=4` is the default recurrent depth.
- GRPO with DAPO-style loss is the only RL baseline in phase 1.
- AIME24 is optional because of variance and budget.
- Adaptive early exit is deferred until fixed-depth looped models show a real signal.
