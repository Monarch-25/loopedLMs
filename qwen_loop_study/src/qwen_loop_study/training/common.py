"""Shared training helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_loop_study.compat import configure_transformers_runtime
from qwen_loop_study.config import DataSpec, ModelSpec, TrainingSpec
from qwen_loop_study.models import LoopedQwenForCausalLM


def prepare_runtime_environment() -> None:
    configure_transformers_runtime()


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name in {"auto", "", None}:
        return "auto"
    return getattr(torch, dtype_name)


def load_tokenizer(model_spec: ModelSpec):
    tokenizer_name = model_spec.tokenizer_name_or_path or model_spec.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_spec.revision,
        trust_remote_code=model_spec.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return tokenizer


def build_model(model_spec: ModelSpec):
    model_kwargs = {
        "revision": model_spec.revision,
        "trust_remote_code": model_spec.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(model_spec.torch_dtype),
    }
    if model_spec.attn_implementation:
        model_kwargs["attn_implementation"] = model_spec.attn_implementation
    if model_spec.architecture == "dense":
        return AutoModelForCausalLM.from_pretrained(model_spec.model_name_or_path, **model_kwargs)
    config_path = Path(model_spec.model_name_or_path) / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        if saved_config.get("model_type") == "looped_qwen":
            return LoopedQwenForCausalLM.from_pretrained(model_spec.model_name_or_path, **model_kwargs)
    return LoopedQwenForCausalLM.from_dense_pretrained(
        model_spec.model_name_or_path,
        num_unique_layers=model_spec.num_unique_layers or 1,
        recurrent_steps=model_spec.recurrent_steps,
        enable_exit_head=model_spec.enable_exit_head,
        early_exit_threshold=model_spec.early_exit_threshold,
        **model_kwargs,
    )


def load_prepared_or_raw_dataset(spec: DataSpec) -> DatasetDict:
    if spec.dataset_path_override:
        return load_from_disk(spec.dataset_path_override)
    return load_dataset(spec.dataset_name, name=spec.dataset_config_name)


def build_training_kwargs(training: TrainingSpec, *, remove_unused_columns: bool = False) -> dict[str, Any]:
    return {
        "output_dir": training.output_dir,
        "seed": training.seed,
        "per_device_train_batch_size": training.per_device_train_batch_size,
        "per_device_eval_batch_size": training.per_device_eval_batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "warmup_ratio": training.warmup_ratio,
        "max_steps": training.max_steps,
        "num_train_epochs": training.num_train_epochs,
        "logging_steps": training.logging_steps,
        "eval_steps": training.eval_steps,
        "save_steps": training.save_steps,
        "save_total_limit": training.save_total_limit,
        "gradient_checkpointing": training.gradient_checkpointing,
        "bf16": training.bf16,
        "fp16": training.fp16,
        "use_cpu": training.use_cpu,
        "use_mps_device": training.use_mps_device,
        "report_to": training.report_to or [],
        "disable_tqdm": training.disable_tqdm,
        "logging_first_step": training.logging_first_step,
        "dataloader_num_workers": training.dataloader_num_workers,
        "remove_unused_columns": remove_unused_columns,
        "save_safetensors": True,
        "logging_strategy": "steps",
    }


def append_manifest(output_dir: str, payload: dict[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_summary_csv(path: str, row: dict[str, Any]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_metrics(output_dir: str, split: str, metrics: dict[str, Any]) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"{split}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
