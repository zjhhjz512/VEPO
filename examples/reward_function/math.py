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

def _get_diff_tensor(old_log_probs, aug_log_probs):
    if old_log_probs is None or aug_log_probs is None:
        return None
    if old_log_probs.numel() == 0 or aug_log_probs.numel() == 0:
        return None
    if old_log_probs.shape != aug_log_probs.shape:
        return None
    old_flat = old_log_probs.reshape(-1).float()
    aug_flat = aug_log_probs.reshape(-1).float()
    diff = old_flat - aug_flat
    return torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def visual_dependence_score(diff: torch.Tensor | None, threshold: float) -> float:
    try:
        if diff is None or diff.numel() == 0:
            return 0.0
        safe_diff = torch.nan_to_num(diff.float(), nan=0.0, posinf=0.0, neginf=0.0)
        exceed_sum = torch.relu(safe_diff - threshold).sum().item()
        return exceed_sum / float(diff.numel())
    except Exception as e:
        print(f"[Warning] visual_dependence_score failed: {e}")
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


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.05,
    perception_quantile: float = 0.6,
    include_perception_in_reward: bool = False,
    perception_weight: float = 0.5,
) -> list[dict[str, float]]:
    scores = []

    perception_quantile = min(max(float(perception_quantile), 0.0), 1.0)

    uids = [reward_input.get("uid") for reward_input in reward_inputs]
    group2indices = defaultdict(list)
    for i, uid in enumerate(uids):
        group2indices[uid if uid is not None else f"__singleton_{i}"].append(i)

    diffs = [
        _get_diff_tensor(reward_input.get("log_probs"), reward_input.get("aug_log_probs"))
        for reward_input in reward_inputs
    ]
    perception_scores = [0.0 for _ in reward_inputs]

    for _, indices in group2indices.items():
        group_diffs = [diffs[i] for i in indices if diffs[i] is not None and diffs[i].numel() > 0]
        if not group_diffs:
            continue

        all_group_tokens = torch.cat(
            [torch.nan_to_num(diff.float(), nan=0.0, posinf=0.0, neginf=0.0) for diff in group_diffs],
            dim=0,
        )
        threshold = float(torch.quantile(all_group_tokens, perception_quantile).item())
        for idx in indices:
            raw_score = visual_dependence_score(diffs[idx], threshold)
            perception_scores[idx] = float(raw_score)

    for i, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        perception_score = perception_scores[i]
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        if include_perception_in_reward:
            overall_score = overall_score + perception_weight * perception_score * accuracy_score

        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": perception_score,
                "visual_dependence": perception_score,
                "visual_adv_gate": accuracy_score,
            }
        )

    return scores