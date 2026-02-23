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
        
        # Count where diff - threshold > 0
        positive_count = ((diff - threshold) > 0).float().sum()
        total_len = max(old_log_probs.numel(), 1.0)
        
        # Log positive_count to file
        try:
            with open("perception_log.txt", "a") as f:
                f.write(f"{positive_count.item()}\n")
        except Exception as e:
            print(f"[Warning] Failed to log positive_count: {e}")
        
        # gain = (positive_count * 15 / total_len).item()

        reward = (positive_count * 20 / total_len).item()
        
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
