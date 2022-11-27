/bin/sh

python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name Naive-lr0_005 --cuda 0
python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name Naive-lr0_005 --cuda 0
