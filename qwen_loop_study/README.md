# Qwen Loop Study

Research scaffold for the Qwen-only study:

- Dense vs post-hoc looped `Qwen/Qwen2.5-0.5B`
- Dense vs post-hoc looped `Qwen/Qwen2.5-1.5B`
- Recovery -> SFT -> GRPO -> evaluation

## Layout

- `src/qwen_loop_study/models/looped_qwen.py`: shared-block Qwen conversion
- `src/qwen_loop_study/data/build_splits.py`: deterministic split building and decontamination
- `src/qwen_loop_study/training/recover.py`: dense-teacher recovery stage
- `src/qwen_loop_study/training/sft.py`: math SFT entrypoint
- `src/qwen_loop_study/training/grpo.py`: GRPO entrypoint with verifiable rewards
- `src/qwen_loop_study/eval/run_eval.py`: benchmark runner and artifact writer
- `configs/`: frozen run configs per condition
- `reports/qwen_looplm_math_study.md`: report template

## Environment

The package is structured to support two environments:

- `lockfiles/local-smoke.txt`: tiny-model correctness on local MPS/CPU
- `lockfiles/gpu.txt`: intended CUDA stack for real runs

## Typical flow

```bash
PYTHONPATH=src qwen-loop-build-splits --config configs/sft/sft_dense_0p5b.yaml
PYTHONPATH=src qwen-loop-recover --config configs/recovery/recover_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-sft --config configs/sft/sft_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-grpo --config configs/grpo/grpo_looped_0p5b.yaml
PYTHONPATH=src qwen-loop-eval --config configs/eval/eval_looped_0p5b.yaml
```

## Notes

- This code studies post-hoc loopification of pretrained Qwen, not from-scratch Ouro pretraining.
- Adaptive early exit is scaffolded in the model code but intentionally disabled in the default configs.
- All training/eval entrypoints are written so tiny synthetic runs can execute locally before scaling out.
