#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python experiments/split_sub_vqa/naive_novel.py --mode novel_test --color_attri --exp_name color-GEM-lr0_01-p32-m0_3 --cuda 0
CUDA_VISIBLE_DEVICES=6 python experiments/split_sub_vqa/naive_novel.py --mode non_novel_test --color_attri --exp_name color-GEM-lr0_01-p32-m0_3 --cuda 0
