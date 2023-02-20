#!/bin/sh
export WANDB_MODE=offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}
CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --exp_name vit_L_lr1e-5 --eval_patience 20 --project_name 'CGQA' --dataset 'cgqa' --return_task_id --use_wandb --model_backbone vit --learning_rate 0.00001 & # > ../avalanche-experiments/out/test.out 2>&1
CUDA_VISIBLE_DEVICES=1 python3 experiments/continual_training.py --exp_name vit_L_lr1e-4 --eval_patience 20 --project_name 'CGQA' --dataset 'cgqa' --return_task_id --use_wandb --model_backbone vit --learning_rate 0.0001 &
CUDA_VISIBLE_DEVICES=2 python3 experiments/continual_training.py --exp_name vit_L_lr1e-3 --eval_patience 20 --project_name 'CGQA' --dataset 'cgqa' --return_task_id --use_wandb --model_backbone vit --learning_rate 0.001

echo FINISH!