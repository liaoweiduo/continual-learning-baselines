#!/bin/bash

#python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name ER-lr0_01-m1000 --cuda 1
#python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name ER-lr0_01-m1000 --cuda 1

#CUDA_VISIBLE_DEVICES=1 python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name ER-lr0_01-m5000
#CUDA_VISIBLE_DEVICES=1 python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name ER-lr0_01-m5000

# freeze FE
CUDA_VISIBLE_DEVICES=1 python experiments/split_sys_vqa/naive_novel.py --freeze --mode novel_test --exp_name ER-lr0_01-m1000
CUDA_VISIBLE_DEVICES=1 python experiments/split_sys_vqa/naive_novel.py --freeze --mode non_novel_test --exp_name ER-lr0_01-m1000
