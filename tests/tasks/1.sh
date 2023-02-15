#!/bin/sh
export WANDB_MODE=offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}
CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --project_name 'CGQA' --dataset 'cgqa' --model_backbone vit > tests/tasks/test.out 2>&1