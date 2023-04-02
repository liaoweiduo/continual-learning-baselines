#!/bin/sh
export WANDB_MODE=online     # online, offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}
#CUDA_VISIBLE_DEVICES=0 python3 experiments/split_cifar100/continual_training.py --project_name 'SCIFAR100' --dataset 'scifar100' --image_size 32 --n_experiences 20 --use_wandb --model_backbone vit --vit_patch_size 4 --vit_dim 256 --vit_depth 6 --vit_mlp_dim 256 --vit_dropout 0.0 --vit_emb_dropout 0.0 --learning_rate 0.0001 --lr_schedule cos --epochs 200
#CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --exp_name our --use_wandb --strategy our --train_num_exp 1 --use_interactive_logger --epochs 200 --disable_early_stop
#CUDA_VISIBLE_DEVICES=7 python3 experiments/visualize_image_with_similarity_mask.py --exp_name random_model --use_wandb --strategy our --use_interactive_logger
# MNt1_lr_reg2-our-tsk-lr0_005-reg1; random_model
CUDA_VISIBLE_DEVICES=7 python3 experiments/continual_training.py --exp_name concept --use_wandb --strategy concept --use_interactive_logger --multi_concept_weight 1.

echo FINISH!