#!/bin/bash

set -x

CUDA_IDS=3,7
N_GPU=2

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.85

TOTAL_EPOCHES=2

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
EXP_NAME="qwen2_5_vl_3b__grpo__ep${TOTAL_EPOCHES}_$(date +%Y%m%d_%H%M%S)"
# EXP_NAME="qwen2_5_vl_3b__grpo__ep2_20260413_161729"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/baidu.zijing.he/data/VIRL_GEO.parquet \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=2