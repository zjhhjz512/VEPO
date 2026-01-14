#!/bin/bash

set -x

CUDA_IDS=3,4
N_GPU=2

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# MODEL_PATH=Qwen/Qwen2-VL-2B-Instruct

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/baidu.zijing.he/data/r1-vl-5000-sample.parquet \
    data.val_files=/data/baidu.zijing.he/data/r1-vl-5000-test.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_grpo_try \
    trainer.n_gpus_per_node=2
