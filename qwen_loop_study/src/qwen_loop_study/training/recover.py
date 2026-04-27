"""Dense-teacher recovery training for dense and looped students."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from qwen_loop_study.config import RecoveryRunConfig, load_yaml_config
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
    load_tokenizer,
    prepare_runtime_environment,
    save_metrics,
)


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, ce_weight: float, kl_weight: float, temperature: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.temperature = temperature
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        student_logits = outputs.logits.float()
        ce_loss = outputs.loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
            )
        teacher_logits = teacher_outputs.logits.float()
        mask = labels.ne(-100)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl = (kl * mask).sum() / mask.sum().clamp_min(1)
        loss = self.ce_weight * ce_loss + self.kl_weight * (self.temperature**2) * kl
        return (loss, outputs) if return_outputs else loss


def _load_streaming_texts(dataset_name: str, dataset_config: str | None, split: str, text_column: str, max_rows: int | None):
    if Path(dataset_name).exists():
        disk_dataset = load_from_disk(dataset_name)
        dataset = disk_dataset[split] if split in disk_dataset else disk_dataset
    else:
        dataset = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    rows = []
    for index, row in enumerate(dataset):
        rows.append(str(row[text_column]))
        if max_rows is not None and index + 1 >= max_rows:
            break
    return rows


def _load_math_texts(config: RecoveryRunConfig) -> list[str]:
    if Path(config.recovery.math_dataset_name).exists():
        disk_dataset = load_from_disk(config.recovery.math_dataset_name)
        ds = disk_dataset[config.recovery.math_split] if config.recovery.math_split in disk_dataset else disk_dataset
    else:
        ds = load_dataset(
            config.recovery.math_dataset_name,
            name=config.recovery.math_dataset_config,
            split=config.recovery.math_split,
        )
    texts: list[str] = []
    for row in ds:
        prompt = str(row.get(config.data.prompt_column, row.get("problem", row.get("question", ""))))
        solution = str(row.get(config.data.solution_column, row.get("solution", row.get("response", ""))))
        texts.append(f"{prompt}\n\n{solution}".strip())
        if config.recovery.max_stream_rows and len(texts) >= config.recovery.max_stream_rows:
            break
    return texts


def build_recovery_dataset(config: RecoveryRunConfig, tokenizer) -> Dataset:
    fineweb_texts = _load_streaming_texts(
        config.recovery.fineweb_dataset_name,
        config.recovery.fineweb_dataset_config,
        config.recovery.fineweb_split,
        config.data.text_column,
        config.recovery.max_stream_rows,
    )
    math_texts = _load_math_texts(config)
    texts = fineweb_texts + math_texts
    token_budget = 0
    chunks: list[list[int]] = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        token_budget += len(tokens)
        for start in range(0, len(tokens), config.recovery.chunk_length):
            chunk = tokens[start : start + config.recovery.chunk_length]
            if len(chunk) < 2:
                continue
            chunks.append(chunk)
        if token_budget >= config.recovery.max_total_tokens:
            break
    return Dataset.from_dict({"input_ids": chunks})


def tokenize_recovery_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def mapper(batch):
        input_ids = []
        attention_masks = []
        labels = []
        for chunk in batch["input_ids"]:
            trimmed = chunk[:max_length]
            mask = [1] * len(trimmed)
            input_ids.append(trimmed)
            attention_masks.append(mask)
            labels.append(trimmed[:])
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    return dataset.map(mapper, batched=True, remove_columns=dataset.column_names)


def run_recovery(config: RecoveryRunConfig):
    prepare_runtime_environment()
    tokenizer = load_tokenizer(config.model)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.recovery.teacher_model_name_or_path,
        revision=config.recovery.teacher_revision,
        torch_dtype=None if config.model.torch_dtype == "auto" else getattr(torch, config.model.torch_dtype),
    )
    if config.training.gradient_checkpointing:
        teacher_model.config.use_cache = False

    student_model = build_model(config.model)

    # --- W&B: initialize run and log pre-training diagnostics ---
    wandb_run = maybe_init_wandb(
        stage="recovery",
        model=student_model,
        model_spec=config.model,
        data_spec=config.data,
        training_spec=config.training,
        extra_config={
            "recovery": {
                "teacher": config.recovery.teacher_model_name_or_path,
                "ce_weight": config.recovery.ce_weight,
                "kl_weight": config.recovery.kl_weight,
                "temperature": config.recovery.temperature,
                "max_total_tokens": config.recovery.max_total_tokens,
            }
        },
    )

    raw_dataset = build_recovery_dataset(config, tokenizer)
    tokenized = tokenize_recovery_dataset(raw_dataset, tokenizer, config.recovery.chunk_length)
    split = tokenized.train_test_split(test_size=min(max(1, len(tokenized) // 10), 128), seed=config.training.seed)
    training_args = TrainingArguments(
        **build_training_kwargs(config.training, remove_unused_columns=False),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loop_cb = make_loop_callback(student_model, wandb_run)
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        ce_weight=config.recovery.ce_weight,
        kl_weight=config.recovery.kl_weight,
        temperature=config.recovery.temperature,
        args=training_args,
        data_collator=data_collator,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        callbacks=[loop_cb] if loop_cb else None,
    )
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    metrics = dict(train_result.metrics)
    metrics["parameter_count"] = count_parameters(student_model)
    save_metrics(config.training.output_dir, "train", metrics)
    eval_metrics = trainer.evaluate()
    save_metrics(config.training.output_dir, "eval", eval_metrics)
    append_manifest(
        config.training.output_dir,
        {
            "stage": "recovery",
            "model": config.model.model_name_or_path,
            "architecture": config.model.architecture,
            "training": config.training.output_dir,
            "metrics": metrics,
        },
    )

    # --- W&B: log final metrics and checkpoint artifact ---
    log_wandb_metrics(wandb_run, metrics, prefix="recovery")
    log_wandb_metrics(wandb_run, eval_metrics, prefix="recovery/eval")
    log_wandb_artifact(
        wandb_run,
        name=f"recovery-{config.model.architecture}-checkpoint",
        artifact_type="model",
        paths=[
            str(Path(config.training.output_dir) / "model.safetensors"),
            str(Path(config.training.output_dir) / "config.json"),
        ],
    )
    finish_wandb_run(wandb_run, summary_updates={**metrics, **eval_metrics})

    return trainer, metrics, eval_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense-teacher recovery training.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_yaml_config(args.config, RecoveryRunConfig)
    run_recovery(config)


if __name__ == "__main__":
    main()
