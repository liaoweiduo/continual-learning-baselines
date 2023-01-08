#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python experiments/split_sys_vqa/naive_novel.py --mode novel_test --exp_name Naive-lr0_005
#CUDA_VISIBLE_DEVICES=0 python experiments/split_sys_vqa/naive_novel.py --mode non_novel_test --exp_name Naive-lr0_005

# freeze FE
CUDA_VISIBLE_DEVICES=0 python experiments/split_sys_vqa/naive_novel.py --freeze --mode novel_test --exp_name Naive-lr0_005
CUDA_VISIBLE_DEVICES=0 python experiments/split_sys_vqa/naive_novel.py --freeze --mode non_novel_test --exp_name Naive-lr0_005
