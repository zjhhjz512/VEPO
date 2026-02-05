# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
import torch.nn.functional as F
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

@torch.no_grad()
def perception_reward(old_log_probs, aug_log_probs) -> float:
    try:
        # Placeholder for perception reward logic
        if old_log_probs is None or aug_log_probs is None:
            return 0.0
        
        if old_log_probs.numel() == 0 or aug_log_probs.numel() == 0:
            return 0.0

        
        # orig_probs = torch.exp(old_log_probs)
        diff = old_log_probs - aug_log_probs
        
        threshold = 1.0  # Example threshold
        gains = torch.clamp(diff - threshold, min=0.0)
        # gains = F.softplus(gains, beta=5)
        penalties = torch.clamp(diff, max=0.0)
        
        # Use sum and count instead of mean to avoid NaN on empty
        # Also ensure we're doing float division
        # count = max(penalties.numel(), 1)
        # reward_score = gains.sum() + 0.5 * penalties.mean()
        reward_score = gains.sum()
        
        reward = torch.tanh(reward_score * 0.5).item()  # Scale factor
        
        return reward
    except Exception as e:
        print(f"[Warning] perception_reward failed: {e}")
        return 0.0



def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.05, perception_weight: float = 0.5) -> list[dict[str, float]]:
    scores = []
    
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        perception_score = perception_reward(
            reward_input.get("log_probs"),
            reward_input.get("aug_log_probs"),
        )

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + perception_weight * perception_score * accuracy_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": perception_score,
            }
        )

    return scores
