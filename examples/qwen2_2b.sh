#!/bin/bash

set -x

CUDA_IDS=6,7
N_GPU=2

TOTAL_EPOCHES=2

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
EXP_NAME="qwen2_5_vl_3b__grpo__ep${TOTAL_EPOCHES}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/baidu.zijing.he/data/VIRL_GEO.parquet \
    data.val_files=PAPOGalaxy/PAPO_MMK12_test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=2