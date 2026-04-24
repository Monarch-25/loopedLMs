# Qwen LoopLM Math Study

## Research Question

Does post-hoc loopification of pretrained Qwen2.5 match or beat dense Qwen2.5 on math after matched recovery, SFT, and GRPO budgets?

## Scope Note

This is not a faithful Ouro replication. The study converts pretrained dense Qwen checkpoints into looped/shared-block models and then measures whether the architecture can recover and benefit from math post-training.

## Conditions

- Dense `Qwen/Qwen2.5-0.5B`
- Looped `Qwen/Qwen2.5-0.5B` with 6 unique blocks and `R=4`
- Dense `Qwen/Qwen2.5-1.5B`
- Looped `Qwen/Qwen2.5-1.5B` with 7 unique blocks and `R=4`

## Datasets

- Recovery: `HuggingFaceFW/fineweb-edu` + raw text from `open-r1/OpenR1-Math-220k`
- SFT: `open-r1/OpenR1-Math-220k`
- RL: `open-r1/Big-Math-RL-Verified-Processed`
- Eval: `HuggingFaceH4/MATH-500`, `openai/gsm8k`

## Required Tables

1. Stage-by-stage accuracy:
   - recovered
   - SFT
   - GRPO
2. Compute table:
   - train steps
   - wall-clock
   - estimated FLOPs
   - tokens/sec
3. Quality table:
   - exact match
   - bootstrap CI
   - parse rate
   - average completion length
   - average loop steps

## Required Plots

- Accuracy vs compute
- Dense vs looped by stage
- GRPO reward curves
- Loop-step histogram
- Parse-rate comparison

## Failure Analysis

- Recovery instability after surgery
- SFT collapse or formatting failure
- GRPO degradation on looped models
- Cases where dense RL improves but looped RL does not

## Conclusion Template

- Did loopification recover after surgery?
- Did looped Qwen beat dense Qwen after SFT?
- Did looped Qwen benefit from GRPO as much as dense Qwen?
- What should be tried next: adaptive exit, longer recovery, or different RL objectives?
