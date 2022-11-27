/bin/sh

python experiments/split_sub_vqa/naive_novel.py --mode novel_test --color_attri --exp_name color-Naive-lr0_005 --cuda 4
python experiments/split_sub_vqa/naive_novel.py --mode non_novel_test --color_attri --exp_name color-Naive-lr0_00 --cuda 0
