"""Math SFT entrypoint."""

from __future__ import annotations

import argparse

from transformers import PreTrainedTokenizerBase

from qwen_loop_study.config import SFTRunConfig, load_yaml_config
from qwen_loop_study.data.build_splits import ensure_boxed_solution, format_math_prompt
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


def build_sft_text(example, tokenizer: PreTrainedTokenizerBase, config: SFTRunConfig) -> dict:
    prompt = format_math_prompt(example["problem"], require_boxed_answer=config.sft.require_boxed_answer)
    solution = ensure_boxed_solution(example["solution"], example.get("answer"))
    messages = []
    if config.sft.system_prompt:
        messages.append({"role": "system", "content": config.sft.system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": solution})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def run_sft(config: SFTRunConfig):
    prepare_runtime_environment()
    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(config.model)
    model = build_model(config.model)
    dataset = load_prepared_or_raw_dataset(config.data)
    train_split = dataset[config.data.train_split]
    eval_split = dataset[config.data.eval_split]
    train_split = train_split.map(lambda example: build_sft_text(example, tokenizer, config))
    eval_split = eval_split.map(lambda example: build_sft_text(example, tokenizer, config))

    training_args = SFTConfig(
        **build_training_kwargs(config.training, remove_unused_columns=True),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        max_seq_length=config.training.max_seq_length,
        dataset_text_field="text",
        packing=config.sft.packing,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    metrics = dict(train_result.metrics)
    metrics["parameter_count"] = count_parameters(model)
    save_metrics(config.training.output_dir, "train", metrics)
    eval_metrics = trainer.evaluate()
    save_metrics(config.training.output_dir, "eval", eval_metrics)
    append_manifest(
        config.training.output_dir,
        {
            "stage": "sft",
            "model": config.model.model_name_or_path,
            "architecture": config.model.architecture,
            "training": config.training.output_dir,
            "metrics": metrics,
        },
    )
    return trainer, metrics, eval_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT for dense or looped Qwen.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_yaml_config(args.config, SFTRunConfig)
    run_sft(config)


if __name__ == "__main__":
    main()
