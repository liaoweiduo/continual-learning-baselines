#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python experiments/split_sub_vqa/naive_novel.py --mode novel_test --color_attri --exp_name color-LwF-lr0_005-a10-t2 --cuda 0
CUDA_VISIBLE_DEVICES=7 python experiments/split_sub_vqa/naive_novel.py --mode non_novel_test --color_attri --exp_name color-LwF-lr0_005-a10-t2 --cuda 0
