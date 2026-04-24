from __future__ import annotations

import pytest

pytest.importorskip("math_verify")

from qwen_loop_study.config import RewardSpec
from qwen_loop_study.training.grpo import build_reward_function


def test_math_reward_exact_and_boxed():
    reward_fn = build_reward_function(RewardSpec(exact_match_reward=1.0, boxed_parse_reward=0.05, incorrect_reward=0.0))
    rewards = reward_fn(
        completions=[
            [{"content": "Work...\nFinal answer: \\boxed{5}"}],
            [{"content": "Work...\nFinal answer: \\boxed{7}"}],
            [{"content": "No box 7"}],
        ],
        solution=["5", "5", "5"],
    )
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.05)
    assert rewards[2] == pytest.approx(0.0)
