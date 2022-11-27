/bin/sh

python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name ER-lr0_01-m1000 --cuda 1
python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name ER-lr0_01-m1000 --cuda 1
