#!/bin/sh
export WANDB_MODE=online     # online, offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}
#CUDA_VISIBLE_DEVICES=0 python3 experiments/split_cifar100/continual_training.py --project_name 'SCIFAR100' --dataset 'scifar100' --image_size 32 --n_experiences 20 --use_wandb --model_backbone vit --vit_patch_size 4 --vit_dim 256 --vit_depth 6 --vit_mlp_dim 256 --vit_dropout 0.0 --vit_emb_dropout 0.0 --learning_rate 0.0001 --lr_schedule cos --epochs 200
#CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --exp_name our_test --use_wandb --strategy our --model_backbone vit --image_size 224 --train_num_exp 1 --use_interactive_logger --epochs 300 --ssc 1 --scc 1 --learning_rate 1e-5 --skip_fewshot_testing --eval_every 10 --eval_patience 50
#CUDA_VISIBLE_DEVICES=7 python3 experiments/visualize_image_with_similarity_mask.py --exp_name random_model --use_wandb --strategy our --use_interactive_logger
# MNt1_lr_reg2-our-tsk-lr0_005-reg1; random_model
#CUDA_VISIBLE_DEVICES=7 python3 experiments/continual_training.py --exp_name concept --use_wandb --strategy concept --use_interactive_logger --multi_concept_weight 1.

# naive vit (128img+4layer) 1 task
#CUDA_VISIBLE_DEVICES=7 python3 experiments/continual_training.py --exp_name naive-vit-small-1task --strategy naive --use_interactive_logger --use_text_logger --train_num_exp 1 --model_backbone vit --image_size 128 --vit_depth 4 --learning_rate 0.0001 --lr_schedule cos --epochs 200 --skip_fewshot_testing


# er test
#CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --exp_name er_test_3 --use_wandb --strategy er --use_interactive_logger --train_num_exp 3 --skip_fewshot_testing

# multi_task test
#CUDA_VISIBLE_DEVICES=0 python3 experiments/multi_task_training.py --exp_name mt_test --return_task_id --strategy naive --use_interactive_logger

# CAM
# naive-tsk-lr0_008; naive-cls-lr0_003; MT-naive-tsk_True-lr0_001; MT-naive-tsk_False-lr0_005
#for mode in 'sys' 'pro' 'sub' 'non' 'noc'
#do
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --dataset_mode $mode --exp_name naive-tsk-lr0_008 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --dataset_mode $mode --exp_name naive-cls-lr0_003 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --dataset_mode $mode --exp_name MT-naive-tsk_True-lr0_001 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --dataset_mode $mode --exp_name MT-naive-tsk_False-lr0_005 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#done
#for mode in 'nonf' 'nono' 'sysf' 'syso'
#do
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --train_class_order fixed --test_n_way 2 --dataset_mode $mode --exp_name naive-tsk-lr0_008 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --train_class_order fixed --test_n_way 2 --dataset_mode $mode --exp_name naive-cls-lr0_003 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --train_class_order fixed --test_n_way 2 --dataset_mode $mode --exp_name MT-naive-tsk_True-lr0_001 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#CUDA_VISIBLE_DEVICES=0 python3 experiments/fewshot_testing.py --tag CAM --train_class_order fixed --test_n_way 2 --dataset_mode $mode --exp_name MT-naive-tsk_False-lr0_005 --strategy naive --epochs 20 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --eval_every -1 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
#done

# scifar ewc class-il lr 0.001, lambda 1
#CUDA_VISIBLE_DEVICES=0 python3 experiments/continual_training.py --learning_rate '0.001' --ewc_lambda '1' --project_name 'SCIFAR100' --dataset 'scifar100' --use_text_logger --tag 'HT' --strategy 'ewc' --model_backbone 'resnet18' --image_size '32' --epochs '100' --skip_fewshot_testing --exp_name 'HT-ewc-tsk_False-lr0_001-lambda1-rerun'
#CUDA_VISIBLE_DEVICES=1 python3 experiments/continual_training.py --learning_rate '0.001' --ewc_lambda '1' --project_name 'SCIFAR100' --dataset 'scifar100' --use_text_logger --tag 'HT' --strategy 'ewc' --model_backbone 'resnet18' --image_size '32' --epochs '100' --skip_fewshot_testing --disable_early_stop --eval_every -1 --exp_name 'HT-ewc-tsk_False-lr0_001-lambda1-rerun-noearlystop'

#icarl
#CUDA_VISIBLE_DEVICES=1 python3 experiments/split_cifar100/icarl.py

# CAM COBJ
# 'HT-MT-naive-tsk_True-lr0_00231', 'HT-MT-naive-tsk_False-lr0_00123',
# 'HT-naive-tsk_True-lr0_001', 'HT-naive-tsk_False-lr0_001',
#  --eval_every -1
for mode in 'sys' 'pro' 'non' 'noc'
do
CUDA_VISIBLE_DEVICES=7 python3 experiments/fewshot_testing.py --test_n_way 3 --tag CAM --dataset_mode $mode --exp_name HT-10tasks-naive-tsk_True-lr1e-05 --project_name COBJ --dataset cobj --strategy naive --epochs 100 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
CUDA_VISIBLE_DEVICES=7 python3 experiments/fewshot_testing.py --test_n_way 3 --tag CAM --dataset_mode $mode --exp_name HT-10tasks-naive-tsk_False-lr0_001 --project_name COBJ --dataset cobj --strategy naive --epochs 100 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
CUDA_VISIBLE_DEVICES=7 python3 experiments/fewshot_testing.py --test_n_way 3 --tag CAM --dataset_mode $mode --exp_name HT-MT-naive-tsk_True-lr0_00231 --project_name COBJ --dataset cobj --strategy naive --epochs 100 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
CUDA_VISIBLE_DEVICES=7 python3 experiments/fewshot_testing.py --test_n_way 3 --tag CAM --dataset_mode $mode --exp_name HT-MT-naive-tsk_False-lr0_00123 --project_name COBJ --dataset cobj --strategy naive --epochs 100 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb
done
#CUDA_VISIBLE_DEVICES=7 python3 experiments/fewshot_testing.py --tag CAM --dataset_mode sys --exp_name HT-naive-tsk_True-lr0_001 --project_name COBJ --dataset cobj --strategy naive --epochs 100 --test_task_id 20 --use_cam_visualization --learning_rate 0.001 --test_freeze_feature_extractor --ignore_finished_testing --use_wandb


echo FINISH!

