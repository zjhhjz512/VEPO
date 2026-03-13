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
import math
import torch
import torch.nn.functional as F
from typing import Any
from collections import defaultdict

from mathruler.grader import extract_boxed_content, grade_answer
# from verl.trainer.info_gain import InfoGainAdvantageCalculator

# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"

# # 全局单例，避免重复加载模型
# _INFO_GAIN_CALCULATOR = None

# def get_info_gain_calculator():
#     global _INFO_GAIN_CALCULATOR
#     if _INFO_GAIN_CALCULATOR is None:
#         _INFO_GAIN_CALCULATOR = InfoGainAdvantageCalculator(
#             model_name="Qwen/Qwen3-Embedding-0.6B"
#         )
#     return _INFO_GAIN_CALCULATOR


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



# def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.05, perception_weight: float = 0.5, info_gain_weight: float = 0.1) -> list[dict[str, float]]:
#     scores = []
    
#     # 1. 提取所有文本和 uid 用于计算信息增益
#     texts = []
#     uids = []
#     for reward_input in reward_inputs:
#         response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
#         texts.append(response)
#         uids.append(reward_input.get("uid"))
        
#     # 2. 批量计算信息增益分数
#     info_gain_scores = [0.0] * len(texts)
#     # 只有当存在有效的 uid 且 info_gain_weight > 0 时才计算（GRPO 需要根据 uid 分组，验证集通常 weight 为 0）
#     if info_gain_weight > 0 and any(uid is not None for uid in uids):
#         try:
#             calculator = get_info_gain_calculator()
#             # 直接调用 info_gain.py 中的计算方法
#             info_gain_scores_tensor = calculator.compute_info_gain_score(texts, uids)
#             info_gain_scores = info_gain_scores_tensor.tolist()
#         except Exception as e:
#             print(f"[Warning] Info gain calculation failed: {e}")

#     # 3. 计算其他奖励并汇总
#     for i, reward_input in enumerate(reward_inputs):
#         response = texts[i]
#         format_score = format_reward(response)
#         accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        
#         perception_score = perception_reward(
#             reward_input.get("log_probs"),
#             reward_input.get("aug_log_probs"),
#         )
        
#         info_gain_score = math.exp(info_gain_scores[i])

#         # 如果 info_gain_weight 为 0 (例如在验证集或未开启时)，则不乘 info_gain_score
#         if info_gain_weight > 0:
#             overall_score = ((1 - format_weight) * accuracy_score + format_weight * format_score + perception_weight * perception_score * accuracy_score) * info_gain_score
#         else:
#             overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score + perception_weight * perception_score * accuracy_score

#         scores.append(
#             {
#                 "overall": overall_score,
#                 "format": format_score,
#                 "accuracy": accuracy_score,
#                 "perception": perception_score,
#                 "info_gain": info_gain_score,
#             }
#         )

#     return scores


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