#!/bin/bash

set -x

CUDA_IDS=3,4
N_GPU=2

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct
# MODEL_PATH=/data/baidu.zijing.he/code/R1-VL/sft/LlamaFactory/saves/qwen2vl_2b/full/sft_epoch2_lr2e-5_Qwen2-VL-2b 

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=512
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=4096

# Append timestamp to experiment name to prevent overwriting or skipping
EXP_NAME="qwen2_5_vl_3b__grpo__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"

CONGI_FILE="examples/configs/config_try.yaml"
TRAIN_FILE="/data/baidu.zijing.he/data/r1-vl-5000-sample.parquet"
VAL_FILE="/data/baidu.zijing.he/data/r1-vl-5000-test.parquet"

FORMAT_PROMPT="examples/format_prompt/math.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.val_freq=5 \
    trainer.save_freq=5 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH}