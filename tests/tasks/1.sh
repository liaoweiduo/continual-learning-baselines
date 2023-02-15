#!/bin/sh
export WANDB_MODE=offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}
CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --learning_rate '0.03' --use_wandb --project_name 'CGQA' --dataset 'cgqa' --return_task_id --strategy 'er' --exp_name 'er-tsk-lr0_03' >> tests/tasks/test.out 2>&1