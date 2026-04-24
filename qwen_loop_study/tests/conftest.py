from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


@pytest.fixture()
def tiny_tokenizer(tmp_path: Path) -> Path:
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "user": 4,
        "assistant": 5,
        "system": 6,
        "Solve": 7,
        "the": 8,
        "problem": 9,
        "carefully": 10,
        "2": 11,
        "+": 12,
        "3": 13,
        "5": 14,
        "\\boxed": 15,
        "{": 16,
        "}": 17,
        "Answer": 18,
        ":": 19,
        "Final": 20,
        "math": 21,
        "show": 22,
        "reasoning": 23,
        ".": 24,
        "What": 25,
        "is": 26,
        "?": 27,
        "1": 28,
        "4": 29,
        "6": 30,
        "7": 31,
        "8": 32,
        "9": 33,
        "10": 34,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{{ eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    path = tmp_path / "tokenizer"
    hf_tokenizer.save_pretrained(path)
    return path


@pytest.fixture()
def tiny_model_dir(tmp_path: Path, tiny_tokenizer: Path) -> Path:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tiny_tokenizer)
    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen2ForCausalLM(config)
    path = tmp_path / "dense_model"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture()
def tiny_dataset_dir(tmp_path: Path) -> Path:
    train = Dataset.from_dict(
        {
            "problem": ["What is 2 + 3 ?", "What is 1 + 4 ?", "What is 3 + 3 ?"],
            "solution": [
                "Add the numbers.\nFinal answer: \\boxed{5}",
                "Add carefully.\nFinal answer: \\boxed{5}",
                "Add carefully.\nFinal answer: \\boxed{6}",
            ],
            "answer": ["5", "5", "6"],
            "source": ["toy", "toy", "toy"],
            "difficulty": ["medium", "hard", "medium"],
        }
    )
    validation = Dataset.from_dict(
        {
            "problem": ["What is 2 + 2 ?"],
            "solution": ["Add carefully.\nFinal answer: \\boxed{4}"],
            "answer": ["4"],
            "source": ["toy"],
            "difficulty": ["medium"],
        }
    )
    dataset = DatasetDict(train=train, validation=validation, test=validation)
    path = tmp_path / "dataset"
    dataset.save_to_disk(path)
    return path


@pytest.fixture()
def tiny_recovery_corpus_dir(tmp_path: Path) -> Path:
    train = Dataset.from_dict({"text": ["Solve the math problem carefully", "Final answer boxed", "2 + 3 = 5"]})
    dataset = DatasetDict(train=train)
    path = tmp_path / "recovery"
    dataset.save_to_disk(path)
    return path
