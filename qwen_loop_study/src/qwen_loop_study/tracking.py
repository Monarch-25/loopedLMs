"""Weights & Biases helpers for study instrumentation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from qwen_loop_study.config import DataSpec, EvalSpec, ModelSpec, TrainingSpec, WandbSpec


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def wandb_is_enabled(spec: WandbSpec | None, report_to: list[str] | None = None) -> bool:
    if spec and spec.enabled:
        return True
    return "wandb" in (report_to or [])


def _import_wandb(enabled: bool):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in missing wandb envs
        raise ImportError(
            "Weights & Biases logging is enabled but `wandb` is not installed. "
            "Install it in the active environment or disable the wandb block."
        ) from exc
    return wandb


def model_diagnostics(model: torch.nn.Module, model_spec: ModelSpec) -> dict[str, float | int]:
    diagnostics: dict[str, float | int] = {}
    all_params = list(model.parameters())
    diagnostics["model/parameter_count"] = sum(param.numel() for param in all_params)
    diagnostics["model/trainable_parameter_count"] = sum(param.numel() for param in all_params if param.requires_grad)
    diagnostics["model/trainable_fraction"] = (
        diagnostics["model/trainable_parameter_count"] / max(1, diagnostics["model/parameter_count"])
    )
    diagnostics["model/is_looped"] = 1 if model_spec.architecture == "looped" else 0
    diagnostics["model/recurrent_steps"] = model_spec.recurrent_steps
    diagnostics["model/num_unique_layers"] = model_spec.num_unique_layers or 0
    diagnostics["model/early_exit_enabled"] = 1 if model_spec.enable_exit_head else 0
    if model_spec.architecture == "looped":
        virtual_depth = getattr(getattr(model, "model", None), "virtual_depth", 0)
        diagnostics["model/virtual_depth"] = virtual_depth
        diagnostics["model/parameter_compression_ratio"] = (
            virtual_depth / max(1, model_spec.num_unique_layers or virtual_depth)
        )

    component_names = {
        "embed_tokens": getattr(getattr(model, "model", model), "embed_tokens", None),
        "final_norm": getattr(getattr(model, "model", model), "norm", None),
        "lm_head": getattr(model, "lm_head", None),
        "exit_head": getattr(model, "exit_head", None),
    }
    for name, module in component_names.items():
        if module is None:
            continue
        params = [param.detach().float() for param in module.parameters() if param is not None]
        if not params:
            continue
        diagnostics[f"model/{name}_param_norm"] = float(torch.sqrt(sum(param.pow(2).sum() for param in params)))

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers:
        layer_norms = []
        for layer_idx, layer in enumerate(layers):
            params = [param.detach().float() for param in layer.parameters()]
            total_norm = float(torch.sqrt(sum(param.pow(2).sum() for param in params)))
            layer_norms.append(total_norm)
            if len(layers) <= 8 or layer_idx in {0, len(layers) // 2, len(layers) - 1}:
                diagnostics[f"model/layer_{layer_idx:02d}_param_norm"] = total_norm
        diagnostics["model/layer_param_norm_mean"] = float(sum(layer_norms) / len(layer_norms))
        diagnostics["model/layer_param_norm_std"] = float(pd.Series(layer_norms).std(ddof=0))
    return diagnostics


def dataset_diagnostics(dataset, data_spec: DataSpec, split_name: str, sample_size: int = 1024) -> dict[str, float | int]:
    diagnostics: dict[str, float | int] = {}
    if dataset is None:
        return diagnostics
    num_rows = len(dataset)
    diagnostics[f"data/{split_name}_samples"] = num_rows
    if num_rows == 0:
        return diagnostics
    sample = dataset.select(range(min(num_rows, sample_size)))
    prompt_column = "problem" if "problem" in sample.column_names else data_spec.prompt_column
    solution_column = "solution" if "solution" in sample.column_names else data_spec.solution_column
    prompt_lengths = [len(str(item).split()) for item in sample[prompt_column]] if prompt_column in sample.column_names else []
    solution_lengths = [len(str(item).split()) for item in sample[solution_column]] if solution_column in sample.column_names else []
    if prompt_lengths:
        diagnostics[f"data/{split_name}_avg_prompt_tokens"] = float(sum(prompt_lengths) / len(prompt_lengths))
        diagnostics[f"data/{split_name}_max_prompt_tokens"] = max(prompt_lengths)
    if solution_lengths:
        diagnostics[f"data/{split_name}_avg_solution_tokens"] = float(sum(solution_lengths) / len(solution_lengths))
        diagnostics[f"data/{split_name}_max_solution_tokens"] = max(solution_lengths)
    source_column = "source" if "source" in sample.column_names else data_spec.source_column
    if source_column in sample.column_names:
        source_counts = pd.Series(sample[source_column]).value_counts().head(10)
        for source, count in source_counts.items():
            diagnostics[f"data/{split_name}_source/{source}"] = int(count)
    difficulty_column = "difficulty" if "difficulty" in sample.column_names else data_spec.difficulty_column
    if difficulty_column in sample.column_names:
        difficulty_counts = pd.Series(sample[difficulty_column]).value_counts().head(10)
        for difficulty, count in difficulty_counts.items():
            diagnostics[f"data/{split_name}_difficulty/{difficulty}"] = int(count)
    return diagnostics


def _default_run_name(stage: str, output_dir: str, model_spec: ModelSpec) -> str:
    base = Path(output_dir).name
    model_stub = Path(model_spec.model_name_or_path).name.replace("/", "-")
    return f"{stage}-{base}-{model_stub}"


def maybe_init_wandb(
    *,
    stage: str,
    model: torch.nn.Module | None,
    model_spec: ModelSpec,
    data_spec: DataSpec | None,
    training_spec: TrainingSpec | None = None,
    eval_spec: EvalSpec | None = None,
    train_dataset=None,
    eval_dataset=None,
    extra_config: dict[str, Any] | None = None,
):
    wandb_spec = (training_spec.wandb if training_spec else None) or (eval_spec.wandb if eval_spec else None)
    report_to = training_spec.report_to if training_spec else ["wandb"] if wandb_spec and wandb_spec.enabled else []
    enabled = wandb_is_enabled(wandb_spec, report_to)
    wandb = _import_wandb(enabled)
    if not enabled or wandb is None:
        return None

    run = wandb.init(
        project=wandb_spec.project or os.environ.get("WANDB_PROJECT", "qwen-loop-study"),
        entity=wandb_spec.entity or os.environ.get("WANDB_ENTITY"),
        group=wandb_spec.group,
        job_type=wandb_spec.job_type or stage,
        name=wandb_spec.name or _default_run_name(stage, (training_spec.output_dir if training_spec else eval_spec.output_predictions_path), model_spec),
        tags=list(dict.fromkeys((wandb_spec.tags or []) + [stage, model_spec.architecture])),
        notes=wandb_spec.notes,
        mode=wandb_spec.mode,
        dir=(training_spec.output_dir if training_spec else Path(eval_spec.output_predictions_path).parent.as_posix()),
        save_code=wandb_spec.save_code,
        config=_to_serializable(
            {
                "stage": stage,
                "model": model_spec,
                "data": data_spec,
                "training": training_spec,
                "eval": eval_spec,
                "extra": extra_config or {},
            }
        ),
        reinit="finish_previous",
    )
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("train/*", step_metric="trainer/global_step")
    wandb.define_metric("eval/*", step_metric="trainer/global_step")
    wandb.define_metric("recovery/*", step_metric="trainer/global_step")
    wandb.define_metric("grpo/*", step_metric="trainer/global_step")
    wandb.define_metric("stage/*")

    if model is not None:
        run.summary.update(model_diagnostics(model, model_spec))
        if wandb_spec.watch:
            wandb.watch(model, log=wandb_spec.watch_log, log_freq=wandb_spec.watch_log_freq, log_graph=False)
    if data_spec is not None:
        run.summary.update(dataset_diagnostics(train_dataset, data_spec, "train"))
        run.summary.update(dataset_diagnostics(eval_dataset, data_spec, "eval"))
    return run


def log_wandb_metrics(run, metrics: dict[str, Any], prefix: str | None = None, step: int | None = None) -> None:
    if run is None:
        return
    import wandb  # type: ignore

    payload = {
        (f"{prefix}/{key}" if prefix else key): value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool))
    }
    if step is not None:
        payload["trainer/global_step"] = step
    if payload:
        wandb.log(payload)


def log_wandb_table(run, name: str, dataframe: pd.DataFrame) -> None:
    if run is None or dataframe.empty:
        return
    import wandb  # type: ignore

    run.log({name: wandb.Table(dataframe=dataframe)})


def log_wandb_artifact(run, name: str, artifact_type: str, paths: list[str]) -> None:
    if run is None:
        return
    import wandb  # type: ignore

    artifact = wandb.Artifact(name=name, type=artifact_type)
    for path in paths:
        if Path(path).exists():
            artifact.add_file(path)
    run.log_artifact(artifact)


def finish_wandb_run(run, summary_updates: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary_updates:
        run.summary.update(_to_serializable(summary_updates))
    run.finish()


# ---------------------------------------------------------------------------
# Per-virtual-step monitoring for looped models
# ---------------------------------------------------------------------------


class LoopedModelMonitor:
    """Captures per-virtual-step hidden-state norms via forward hooks on shared layers.

    When the looped model's forward pass iterates ``virtual_depth`` times
    through ``num_unique_layers`` physical layers, this monitor records the
    output hidden-state L2 norm at every virtual step.  After the forward
    pass the caller can read ``step_hidden_norms`` (length = virtual_depth)
    and ``step_grad_norms`` (populated after backward).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._inner = getattr(model, "model", None)
        self._layers = getattr(self._inner, "layers", None) if self._inner else None
        self._num_unique = getattr(self._inner, "num_unique_layers", 0) if self._inner else 0
        self._virtual_depth = getattr(self._inner, "virtual_depth", 0) if self._inner else 0

        # Recorded per-virtual-step norms (reset on each forward)
        self.step_hidden_norms: list[float] = []
        # Recorded per-virtual-step gradient norms (reset on each backward)
        self.step_grad_norms: list[float] = []

        self._call_counter: int = 0
        self._grad_accumulator: dict[int, list[float]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    @property
    def is_looped(self) -> bool:
        return self._num_unique > 0 and self._virtual_depth > self._num_unique

    def attach(self) -> "LoopedModelMonitor":
        """Register forward and backward hooks on each physical shared layer."""
        if not self.is_looped or self._layers is None:
            return self

        for layer_idx, layer in enumerate(self._layers):
            # Forward hook: capture hidden-state norm at each virtual step
            def _fwd_hook(module, inp, out, _idx=layer_idx):
                hidden = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hidden):
                    norm = float(hidden.detach().float().norm().item())
                    self.step_hidden_norms.append(norm)

            # Backward hook: capture gradient norm flowing through each virtual step
            def _bwd_hook(module, grad_input, grad_output, _idx=layer_idx):
                grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                if torch.is_tensor(grad):
                    norm = float(grad.detach().float().norm().item())
                    self._grad_accumulator.setdefault(self._call_counter, []).append(norm)

            self._hooks.append(layer.register_forward_hook(_fwd_hook))
            self._hooks.append(layer.register_full_backward_hook(_bwd_hook))
        return self

    def detach(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Clear recorded norms before a new forward/backward pass."""
        self._call_counter += 1
        self.step_hidden_norms = []
        self.step_grad_norms = []
        self._grad_accumulator = {}

    def finalize_grads(self) -> None:
        """Consolidate gradient norms collected during backward into a list."""
        if self._call_counter in self._grad_accumulator:
            # Backward hooks fire in reverse order, so reverse to match forward step ordering
            self.step_grad_norms = list(reversed(self._grad_accumulator[self._call_counter]))

    def snapshot(self) -> dict[str, float]:
        """Return a metrics dict summarising the last forward/backward pass."""
        metrics: dict[str, float] = {}
        if not self.is_looped:
            return metrics

        # Per-virtual-step hidden-state norms
        for step_idx, norm in enumerate(self.step_hidden_norms):
            metrics[f"loop/virtual_step_{step_idx:02d}_hidden_norm"] = norm
        if self.step_hidden_norms:
            metrics["loop/hidden_norm_mean"] = sum(self.step_hidden_norms) / len(self.step_hidden_norms)
            metrics["loop/hidden_norm_std"] = float(
                (sum((v - metrics["loop/hidden_norm_mean"]) ** 2 for v in self.step_hidden_norms)
                 / len(self.step_hidden_norms)) ** 0.5
            )

        # Per-virtual-step gradient norms
        self.finalize_grads()
        for step_idx, norm in enumerate(self.step_grad_norms):
            metrics[f"loop/virtual_step_{step_idx:02d}_grad_norm"] = norm
        if self.step_grad_norms:
            metrics["loop/grad_norm_mean"] = sum(self.step_grad_norms) / len(self.step_grad_norms)
            metrics["loop/grad_norm_std"] = float(
                (sum((v - metrics["loop/grad_norm_mean"]) ** 2 for v in self.step_grad_norms)
                 / len(self.step_grad_norms)) ** 0.5
            )

        # Aggregated per-physical-layer gradient norms
        if self._layers is not None:
            for layer_idx, layer in enumerate(self._layers):
                grads = [p.grad.detach().float() for p in layer.parameters() if p.grad is not None]
                if grads:
                    total = float(torch.sqrt(sum(g.pow(2).sum() for g in grads)))
                    metrics[f"loop/shared_layer_{layer_idx:02d}_grad_norm"] = total
                    # Gradient-to-parameter ratio (relative update magnitude)
                    params = [p.detach().float() for p in layer.parameters()]
                    param_norm = float(torch.sqrt(sum(p.pow(2).sum() for p in params)))
                    if param_norm > 0:
                        metrics[f"loop/shared_layer_{layer_idx:02d}_grad_param_ratio"] = total / param_norm

        return metrics


def make_loop_callback(model: torch.nn.Module, wandb_run):
    """Create a HuggingFace TrainerCallback that logs looped-model diagnostics.

    Returns ``None`` if the model is not a looped architecture or W&B is disabled.
    """
    if wandb_run is None:
        return None

    monitor = LoopedModelMonitor(model)
    if not monitor.is_looped:
        return None
    monitor.attach()

    from transformers import TrainerCallback

    class _WandbLoopCallback(TrainerCallback):
        """Logs per-virtual-step hidden/gradient norms at each logging step."""

        def on_step_begin(self, args, state, control, **kwargs):
            monitor.reset()

        def on_log(self, args, state, control, logs=None, **kwargs):
            snapshot = monitor.snapshot()
            if snapshot:
                log_wandb_metrics(wandb_run, snapshot, step=state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            monitor.detach()

    return _WandbLoopCallback()
