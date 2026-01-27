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

def perception_reward(old_log_probs, aug_log_probs, response_tokens) -> tuple[float, str]:
    # Placeholder for perception reward logic
    if old_log_probs is None or aug_log_probs is None:
        return 0.0, ""
    
    orig_probs = torch.exp(old_log_probs)
    diff = old_log_probs - aug_log_probs
    
    topk = 3
    topk_info = ""
    if len(response_tokens) > 0:
        values, indices = torch.topk(diff, k=min(topk, len(diff)))
        topk_info = f"Top-{topk} Max Diffs:\n"
        for val, idx in zip(values, indices):
            token = response_tokens[idx.item()]
            topk_info += f"  - Token: [{token:10s}] | Diff: {val.item():.4f} (Orig: {old_log_probs[idx].item():.4f}, Aug: {aug_log_probs[idx].item():.4f})\n"

    threshold = 2.0  # Example threshold
    gains = F.softplus(diff - threshold, beta=5)
    weighted_gains = gains * orig_probs
    reward = torch.tanh(weighted_gains.sum() * 0.1).item()  # Scale factor
    
    return reward, topk_info


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.05, perception_weight: float = 0.5) -> list[dict[str, float]]:
    scores = []
    
    # 我们只记录每个 batch 的前几个样本到日志，避免日志文件过大
    num_to_log = 5
    
    for i, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
        perception_score, topk_info = perception_reward(
            reward_input.get("log_probs", torch.tensor(0.0)),
            reward_input.get("aug_log_probs", torch.tensor(0.0)),
            reward_input.get("response_tokens", []),
        )

        if i < num_to_log:
            try:
                with open("response_sample.log", "a", encoding="utf-8") as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"Sample {i} | Accuracy: {accuracy_score} | Perception: {perception_score:.4f}\n")
                    f.write(f"Prompt: {reward_input.get('prompt', 'N/A')}\n")
                    f.write(f"Ground Truth: {reward_input['ground_truth']}\n")
                    f.write(f"Response: {reward_input['response']}\n")
                    if topk_info:
                        f.write(f"\n{topk_info}")
                    f.write("="*80 + "\n")
            except Exception as e:
                print(f"Failed to write log: {e}")

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": perception_score,
            }
        )

    return scores
