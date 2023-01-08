#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python experiments/split_sub_vqa/naive_novel.py --mode novel_test --color_attri --exp_name color-Naive-lr0_005
CUDA_VISIBLE_DEVICES=0 python experiments/split_sub_vqa/naive_novel.py --mode non_novel_test --color_attri --exp_name color-Naive-lr0_00
