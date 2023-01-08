#!/bin/bash

#CUDA_VISIBLE_DEVICES=4 python experiments/split_sys_vqa/naive_novel.py --setting class --epochs 60 --mode novel_test --exp_name LwF-lr0_005-a11-t2
#CUDA_VISIBLE_DEVICES=4 python experiments/split_sys_vqa/naive_novel.py --setting class --epochs 60 --mode non_novel_test --exp_name LwF-lr0_005-a11-t2

# freeze FE
CUDA_VISIBLE_DEVICES=5 python experiments/split_sys_vqa/naive_novel.py --freeze --mode novel_test --exp_name LwF-lr0_005-a11-t2
CUDA_VISIBLE_DEVICES=5 python experiments/split_sys_vqa/naive_novel.py --freeze --mode non_novel_test --exp_name LwF-lr0_005-a11-t2
