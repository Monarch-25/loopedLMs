"""Config dataclasses and YAML loading helpers."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Optional, TypeVar, get_args, get_origin, get_type_hints

import yaml


T = TypeVar("T")


@dataclass
class ModelSpec:
    model_name_or_path: str
    architecture: str = "dense"
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    recurrent_steps: int = 1
    num_unique_layers: int | None = None
    enable_exit_head: bool = False
    early_exit_threshold: float | None = None


@dataclass
class DataSpec:
    dataset_name: str
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    prompt_column: str = "problem"
    solution_column: str = "solution"
    answer_column: str = "answer"
    source_column: str = "source"
    difficulty_column: str = "difficulty"
    train_size: int | None = None
    eval_size: int | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    stratify_length_bins: list[int] | None = None
    decontaminate_with: list[str] | None = None
    output_dir: str | None = None
    dataset_path_override: str | None = None
    seed: int = 42
    num_proc: int | None = None


@dataclass
class TrainingSpec:
    output_dir: str
    seed: int = 42
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_steps: int = -1
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    max_seq_length: int = 1024
    gradient_checkpointing: bool = False
    bf16: bool = False
    fp16: bool = False
    use_cpu: bool = False
    use_mps_device: bool = False
    report_to: list[str] | None = None
    disable_tqdm: bool = True
    logging_first_step: bool = True
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None


@dataclass
class RecoverySpec:
    teacher_model_name_or_path: str
    teacher_revision: str | None = None
    fineweb_dataset_name: str = "HuggingFaceFW/fineweb-edu"
    fineweb_dataset_config: str | None = None
    fineweb_split: str = "train"
    math_dataset_name: str = "open-r1/OpenR1-Math-220k"
    math_dataset_config: str | None = None
    math_split: str = "train"
    chunk_length: int = 512
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    temperature: float = 1.0
    max_total_tokens: int = 12_000_000
    fineweb_token_budget: int = 10_000_000
    math_token_budget: int = 2_000_000
    max_stream_rows: int | None = None


@dataclass
class SFTSpec:
    system_prompt: str | None = None
    packing: bool = False
    require_boxed_answer: bool = True


@dataclass
class RewardSpec:
    exact_match_reward: float = 1.0
    boxed_parse_reward: float = 0.05
    incorrect_reward: float = 0.0


@dataclass
class GRPOSpec:
    beta: float = 0.01
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 0.9
    loss_type: str = "dapo"
    use_vllm: bool = False
    reward: RewardSpec = field(default_factory=RewardSpec)


@dataclass
class EvalSpec:
    benchmarks: list[str]
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    output_predictions_path: str = "results/per_example_predictions.parquet"
    summary_csv_path: str = "results/summary.csv"
    num_bootstrap_samples: int = 1000


@dataclass
class RecoveryRunConfig:
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    recovery: RecoverySpec


@dataclass
class SFTRunConfig:
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    sft: SFTSpec


@dataclass
class GRPORunConfig:
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    grpo: GRPOSpec


@dataclass
class EvalRunConfig:
    model: ModelSpec
    data: DataSpec
    eval: EvalSpec


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Optional, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else Any
    if origin is None and str(annotation).startswith("typing.Optional"):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else Any
    if origin is not None and origin in (list, dict):
        return annotation
    if origin is None and hasattr(annotation, "__args__"):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return args[0]
    return annotation


def _materialize_value(annotation: Any, value: Any) -> Any:
    base = _strip_optional(annotation)
    origin = get_origin(base)
    if value is None:
        return None
    if is_dataclass(base):
        return _load_dataclass(base, value)
    if origin is list:
        item_type = get_args(base)[0] if get_args(base) else Any
        return [_materialize_value(item_type, item) for item in value]
    if origin is dict:
        key_type, value_type = get_args(base) if get_args(base) else (Any, Any)
        return {
            _materialize_value(key_type, key): _materialize_value(value_type, item)
            for key, item in value.items()
        }
    return value


def _load_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        annotation = type_hints.get(field.name, field.type)
        if field.name in data:
            kwargs[field.name] = _materialize_value(annotation, data[field.name])
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise KeyError(f"Missing required config field: {cls.__name__}.{field.name}")
    return cls(**kwargs)


def load_yaml_config(path: str | Path, cls: type[T]) -> T:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return _load_dataclass(cls, payload)
