from __future__ import annotations

from pathlib import Path

import yaml

from qwen_loop_study.eval.run_eval import run_evaluation
from qwen_loop_study.training.grpo import run_grpo
from qwen_loop_study.training.recover import run_recovery
from qwen_loop_study.training.sft import run_sft
from qwen_loop_study.config import EvalRunConfig, GRPORunConfig, RecoveryRunConfig, SFTRunConfig
from qwen_loop_study.config import load_yaml_config


def write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_recovery_stage_and_resume(tmp_path: Path, tiny_model_dir: Path, tiny_recovery_corpus_dir: Path):
    config_path = write_yaml(
        tmp_path / "recover.yaml",
        {
            "model": {
                "model_name_or_path": str(tiny_model_dir),
                "tokenizer_name_or_path": str(tiny_model_dir),
                "architecture": "looped",
                "recurrent_steps": 4,
                "num_unique_layers": 2,
            },
            "data": {
                "dataset_name": str(tiny_recovery_corpus_dir),
                "text_column": "text",
                "prompt_column": "problem",
                "solution_column": "solution",
            },
            "training": {
                "output_dir": str(tmp_path / "recover_out"),
                "use_cpu": True,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "max_steps": 1,
                "logging_steps": 1,
                "eval_steps": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "disable_tqdm": True,
            },
            "recovery": {
                "teacher_model_name_or_path": str(tiny_model_dir),
                "fineweb_dataset_name": str(tiny_recovery_corpus_dir),
                "fineweb_split": "train",
                "math_dataset_name": str(tiny_recovery_corpus_dir),
                "math_split": "train",
                "chunk_length": 16,
                "max_total_tokens": 64,
                "max_stream_rows": 3,
            },
        },
    )
    config = load_yaml_config(config_path, RecoveryRunConfig)
    run_recovery(config)
    checkpoint = next((tmp_path / "recover_out").glob("checkpoint-*"))
    config.training.resume_from_checkpoint = str(checkpoint)
    config.training.max_steps = 2
    run_recovery(config)


def test_sft_and_grpo_and_eval_smoke(tmp_path: Path, tiny_model_dir: Path, tiny_dataset_dir: Path):
    sft_config_path = write_yaml(
        tmp_path / "sft.yaml",
        {
            "model": {
                "model_name_or_path": str(tiny_model_dir),
                "tokenizer_name_or_path": str(tiny_model_dir),
                "architecture": "dense",
            },
            "data": {
                "dataset_name": str(tiny_dataset_dir),
                "dataset_path_override": str(tiny_dataset_dir),
                "train_split": "train",
                "eval_split": "validation",
            },
            "training": {
                "output_dir": str(tmp_path / "sft_out"),
                "use_cpu": True,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "max_steps": 1,
                "logging_steps": 1,
                "eval_steps": 1,
                "save_steps": 1,
                "disable_tqdm": True,
            },
            "sft": {"packing": False, "require_boxed_answer": True},
        },
    )
    sft_config = load_yaml_config(sft_config_path, SFTRunConfig)
    run_sft(sft_config)

    grpo_config_path = write_yaml(
        tmp_path / "grpo.yaml",
        {
            "model": {
                "model_name_or_path": str(tmp_path / "sft_out"),
                "tokenizer_name_or_path": str(tmp_path / "sft_out"),
                "architecture": "dense",
            },
            "data": {
                "dataset_name": str(tiny_dataset_dir),
                "dataset_path_override": str(tiny_dataset_dir),
                "train_split": "train",
                "eval_split": "validation",
            },
            "training": {
                "output_dir": str(tmp_path / "grpo_out"),
                "use_cpu": True,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-5,
                    "max_steps": 1,
                    "logging_steps": 1,
                    "eval_steps": 0,
                    "save_steps": 1,
                    "disable_tqdm": True,
                },
            "grpo": {
                "beta": 0.0,
                "num_generations": 2,
                "max_prompt_length": 32,
                "max_completion_length": 8,
                "temperature": 0.7,
                "use_vllm": False,
                "reward": {
                    "exact_match_reward": 1.0,
                    "boxed_parse_reward": 0.05,
                    "incorrect_reward": 0.0,
                },
            },
        },
    )
    grpo_config = load_yaml_config(grpo_config_path, GRPORunConfig)
    run_grpo(grpo_config)

    eval_config_path = write_yaml(
        tmp_path / "eval.yaml",
        {
            "model": {
                "model_name_or_path": str(tmp_path / "grpo_out"),
                "tokenizer_name_or_path": str(tmp_path / "grpo_out"),
                "architecture": "dense",
            },
            "data": {"dataset_name": str(tiny_dataset_dir)},
            "eval": {
                "benchmarks": [str(tiny_dataset_dir)],
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "batch_size": 1,
                "output_predictions_path": str(tmp_path / "predictions.parquet"),
                "summary_csv_path": str(tmp_path / "summary.csv"),
                "num_bootstrap_samples": 32,
            },
        },
    )
    eval_config = load_yaml_config(eval_config_path, EvalRunConfig)
    _, predictions = run_evaluation(eval_config)
    assert (tmp_path / "predictions.parquet").exists()
    assert (tmp_path / "summary.csv").exists()
    assert not predictions.empty
