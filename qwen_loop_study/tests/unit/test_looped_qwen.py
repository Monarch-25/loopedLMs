from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qwen_loop_study.models.looped_qwen import LoopedQwenForCausalLM, shared_layer_groups


def build_dense_model():
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config)


def test_tied_layer_mapping_is_exact():
    dense = build_dense_model()
    looped = LoopedQwenForCausalLM.from_dense_model(dense, num_unique_layers=2, recurrent_steps=4)
    groups = shared_layer_groups(dense.config.num_hidden_layers, 2)
    for shared_idx, group in enumerate(groups):
        target = looped.model.layers[shared_idx].state_dict()
        for name, tensor in target.items():
            if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue
            expected = torch.stack(
                [dense.model.layers[layer_idx].state_dict()[name].float() for layer_idx in group], dim=0
            ).mean(dim=0)
            assert torch.allclose(tensor.float(), expected, atol=1e-5), name


def test_looped_forward_shape_matches_dense():
    dense = build_dense_model()
    looped = LoopedQwenForCausalLM.from_dense_model(dense, num_unique_layers=2, recurrent_steps=4)
    input_ids = torch.randint(0, 64, (2, 6))
    dense_out = dense(input_ids=input_ids)
    looped_out = looped(input_ids=input_ids)
    assert dense_out.logits.shape == looped_out.logits.shape


def test_fixed_depth_is_deterministic():
    dense = build_dense_model()
    looped = LoopedQwenForCausalLM.from_dense_model(dense, num_unique_layers=2, recurrent_steps=4)
    input_ids = torch.randint(0, 64, (1, 5))
    first = looped(input_ids=input_ids).logits
    second = looped(input_ids=input_ids).logits
    assert torch.allclose(first, second)


def test_gate_probabilities_sum_to_one():
    dense = build_dense_model()
    looped = LoopedQwenForCausalLM.from_dense_model(
        dense,
        num_unique_layers=2,
        recurrent_steps=4,
        enable_exit_head=True,
    )
    input_ids = torch.randint(0, 64, (1, 5))
    outputs = looped(input_ids=input_ids, output_loop_metadata=True)
    assert outputs.exit_distribution is not None
    assert torch.allclose(outputs.exit_distribution.sum(dim=-1), torch.ones(1), atol=1e-5)
