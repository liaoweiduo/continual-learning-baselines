#!/bin/bash

#CUDA_VISIBLE_DEVICES=3 python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name GEM-lr0_01-p32-m0_5
#CUDA_VISIBLE_DEVICES=3 python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name GEM-lr0_01-p32-m0_5

# freeze FE
CUDA_VISIBLE_DEVICES=3 python experiments/split_sys_vqa/naive_novel.py --freeze --mode novel_test --exp_name GEM-lr0_01-p32-m0_5
CUDA_VISIBLE_DEVICES=3 python experiments/split_sys_vqa/naive_novel.py --freeze --mode non_novel_test --exp_name GEM-lr0_01-p32-m0_5
