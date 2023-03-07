#!/bin/sh
#export WANDB_MODE=offline
abs_path=`pwd`
echo abs_path:$abs_path
export PYTHONPATH=${PYTHONPATH}:${abs_path}

# cls
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-cls-lr0_003 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'sys' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-cls-lr0_003 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'pro' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-cls-lr0_003 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'sub' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-cls-lr0_003 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'non' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-cls-lr0_003 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'noc' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20

# tsk
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-tsk-lr0_0008 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'sys' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-tsk-lr0_0008 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'pro' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-tsk-lr0_0008 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'sub' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-tsk-lr0_0008 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'non' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20
CUDA_VISIBLE_DEVICES=1 python3 experiments/fewshot_testing.py --exp_name er-tsk-lr0_0008 --project_name 'CGQA' --dataset 'cgqa' --dataset_mode 'noc' --model_backbone resnet18 --test_freeze_feature_extractor --learning_rate 0.001 --eval_patience 20

echo FINISH!