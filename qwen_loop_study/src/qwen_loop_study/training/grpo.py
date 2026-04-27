"""GRPO training entrypoint with math-verifiable rewards."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from qwen_loop_study.config import GRPORunConfig, RewardSpec, load_yaml_config
from qwen_loop_study.data.build_splits import format_math_prompt
from qwen_loop_study.tracking import (
    finish_wandb_run,
    log_wandb_artifact,
    log_wandb_metrics,
    make_loop_callback,
    maybe_init_wandb,
)
from qwen_loop_study.training.common import (
    append_manifest,
    build_model,
    build_training_kwargs,
    count_parameters,
    load_prepared_or_raw_dataset,
    load_tokenizer,
    prepare_runtime_environment,
    save_metrics,
)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    if isinstance(completion, list):
        return " ".join(str(item) for item in completion)
    return str(completion)


def _boxed_parse_success(text: str) -> bool:
    return bool(re.search(r"\\boxed\s*\{.*?\}", text, flags=re.DOTALL))


def build_reward_function(reward_spec: RewardSpec):
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except Exception:  # pragma: no cover - fallback is for environments without math-verify
        parse = None
        verify = None
        LatexExtractionConfig = None
        NormalizationConfig = None

    def reward_fn(completions, solution, **kwargs):
        rewards: list[float] = []
        for completion, gold in zip(completions, solution):
            text = _extract_completion_text(completion)
            if parse is None or verify is None:
                rewards.append(reward_spec.boxed_parse_reward if _boxed_parse_success(text) else reward_spec.incorrect_reward)
                continue
            gold_parsed = parse(gold, extraction_mode="first_match")
            answer_parsed = parse(
                text,
                extraction_mode="first_match",
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
            )
            if gold_parsed and answer_parsed:
                try:
                    if verify(gold_parsed, answer_parsed):
                        rewards.append(reward_spec.exact_match_reward)
                    else:
                        rewards.append(
                            reward_spec.boxed_parse_reward
                            if _boxed_parse_success(text)
                            else reward_spec.incorrect_reward
                        )
                    continue
                except Exception:
                    pass
            rewards.append(reward_spec.boxed_parse_reward if _boxed_parse_success(text) else reward_spec.incorrect_reward)
        return rewards

    return reward_fn


def build_grpo_prompts(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": format_math_prompt(example["problem"], require_boxed_answer=True)}],
        "solution": example.get("answer") or example.get("solution"),
        "source": example.get("source", "unknown"),
        "difficulty": example.get("difficulty", "unknown"),
    }


def run_grpo(config: GRPORunConfig):
    prepare_runtime_environment()
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = load_tokenizer(config.model)
    model = build_model(config.model)
    dataset = load_prepared_or_raw_dataset(config.data)
    train_split = dataset[config.data.train_split].map(build_grpo_prompts)
    do_eval = config.training.eval_steps > 0 and config.data.eval_split in dataset
    eval_split = dataset[config.data.eval_split].map(build_grpo_prompts) if do_eval else None

    # --- W&B: initialize run and log pre-training diagnostics ---
    wandb_run = maybe_init_wandb(
        stage="grpo",
        model=model,
        model_spec=config.model,
        data_spec=config.data,
        training_spec=config.training,
        train_dataset=train_split,
        eval_dataset=eval_split,
        extra_config={
            "grpo": {
                "beta": config.grpo.beta,
                "num_generations": config.grpo.num_generations,
                "max_completion_length": config.grpo.max_completion_length,
                "temperature": config.grpo.temperature,
                "loss_type": config.grpo.loss_type,
                "exact_match_reward": config.grpo.reward.exact_match_reward,
                "boxed_parse_reward": config.grpo.reward.boxed_parse_reward,
            }
        },
    )

    reward_fn = build_reward_function(config.grpo.reward)
    training_args = GRPOConfig(
        **build_training_kwargs(config.training, remove_unused_columns=False),
        do_train=True,
        do_eval=do_eval,
        eval_strategy="steps" if do_eval else "no",
        max_prompt_length=config.grpo.max_prompt_length,
        max_completion_length=config.grpo.max_completion_length,
        num_generations=config.grpo.num_generations,
        temperature=config.grpo.temperature,
        beta=config.grpo.beta,
        use_vllm=config.grpo.use_vllm,
    )
    loop_cb = make_loop_callback(model, wandb_run)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        processing_class=tokenizer,
        callbacks=[loop_cb] if loop_cb else None,
    )
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    metrics = dict(train_result.metrics)
    metrics["parameter_count"] = count_parameters(model)
    save_metrics(config.training.output_dir, "train", metrics)
    eval_metrics = trainer.evaluate() if do_eval else {}
    if do_eval:
        save_metrics(config.training.output_dir, "eval", eval_metrics)
    append_manifest(
        config.training.output_dir,
        {
            "stage": "grpo",
            "model": config.model.model_name_or_path,
            "architecture": config.model.architecture,
            "training": config.training.output_dir,
            "metrics": metrics,
        },
    )

    # --- W&B: log final metrics and checkpoint artifact ---
    log_wandb_metrics(wandb_run, metrics, prefix="grpo")
    if eval_metrics:
        log_wandb_metrics(wandb_run, eval_metrics, prefix="grpo/eval")
    log_wandb_artifact(
        wandb_run,
        name=f"grpo-{config.model.architecture}-checkpoint",
        artifact_type="model",
        paths=[
            str(Path(config.training.output_dir) / "model.safetensors"),
            str(Path(config.training.output_dir) / "config.json"),
        ],
    )
    finish_wandb_run(wandb_run, summary_updates={**metrics, **eval_metrics})

    return trainer, metrics, eval_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO for dense or looped Qwen.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_yaml_config(args.config, GRPORunConfig)
    run_grpo(config)


if __name__ == "__main__":
    main()
