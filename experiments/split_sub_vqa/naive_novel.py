import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin

from models.resnet import ResNet18, MTResNet18
from models.cnn_128 import CNN128, MTCNN128
from experiments.utils import set_seed, create_default_args, create_experiment_folder
from datasets.cgqa import SplitSubGQA


def naive_novel_ssubvqa_ci(override_args=None):
    """
    Naive algorithm on split substitutivity VQA (novel comb) on class-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 10, 'epochs': 20, 'train_mb_size': 32,
        'eval_every': 2, 'eval_mb_size': 50,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        "freeze": False, "label_map": False, "non_comp": False,
        'use_wandb': False, 'project_name': 'Split_Sub_VQA', 'exp_name': 'Naive',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments'
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root, exp_name=args.exp_name,
        project_name=args.project_name)
    args.exp_name = f'Novel-{args.exp_name}'
    if args.freeze:
        args.exp_name = f'{args.exp_name}-frz'
    if args.label_map:
        args.exp_name = f'{args.exp_name}-lm'
    # if args.only_eval:
    #     args.exp_name = f'{args.exp_name}-oe'

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    if args.label_map:  # map with
        train_classes = [16, 15, 17, 14, 1, 9, 0, 12, 6, 7, 5, 13, 2, 18, 11, 3, 8, 4, 10, 19]
        args.label_map = np.arange(20)
        args.label_map[train_classes] = np.arange(20)
    else:
        args.label_map = None

    # ####################
    # BENCHMARK
    # ####################
    # different seed
    benchmark = SplitSubGQA(n_experiences=args.n_experiences, return_task_id=False, seed=4321, shuffle=True,
                            novel_combination=True, label_map=args.label_map,
                            dataset_root=args.dataset_root, non_comp=args.non_comp)

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    loggers = [interactive_logger, text_logger]
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = avl.logging.WandBLogger(
            project_name=args.project_name, run_name=args.exp_name,
            log_artifacts=True,
            path=checkpoint_path,
            dir=exp_path,
            config=vars(args),
        )
        loggers.append(wandb_logger)

    # ####################
    # EVALUATION PLUGIN
    # ####################
    evaluation_plugin = EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        benchmark=benchmark,
        loggers=loggers)

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes: ", [
            benchmark.original_map_int_label_to_tuple[cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
        ])

        # ####################
        # MODEL
        # ####################
        print("Load trained model.")
        if args.model == "resnet":
            model = ResNet18(initial_out_features=20,
                             pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth'),
                             fix=args.freeze)
        else:
            raise Exception("Un-recognized model structure.")

        # ####################
        # STRATEGY INSTANCE
        # ####################
        cl_strategy = Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            # eval_every=args.eval_every,
            # peval_mode="epoch",
        )

        cl_strategy.train(experience,
                          # [benchmark.test_stream[experience.current_experience]],
                          num_workers=8, pin_memory=False)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream[experience.current_experience],
                                        num_workers=8, pin_memory=False))
        print(results[experience.current_experience])

    print("Final results:")
    print(results)

    # ####################
    # STORE RESULTS
    # ####################
    np.save(os.path.join(exp_path, f'results-{args.exp_name}.npy'), results)

    print('###################################')
    accs = [result['Top1_Acc_Stream/eval_phase/test_stream/Task000'] for result in results]
    print('accs:', accs)
    print(f'Top1_Acc_Stream/eval_phase/test_stream/Task000: {np.mean(accs)} ({np.std(accs)})')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--model", type=str, default='resnet', help="In [resnet, cnn]")
    parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
    parser.add_argument("--exp_name", type=str, default='Naive')
    # parser.add_argument("--setting", type=str, default='task', help="task: Task IL or class: class IL")
    parser.add_argument("--freeze", action='store_true', help="whether freeze feature extractor.")
    parser.add_argument("--label_map", action='store_true',
                        help="whether map novel label to one trained in continual training phase.")
    # parser.add_argument("--only_eval", action='store_true',
    #                     help="whether only do eval and not train.")
    parser.add_argument("--non_comp", action='store_true', help="non compositional dataset")
    args = parser.parse_args()

    res = naive_novel_ssubvqa_ci(vars(args))


    '''
    export PYTHONPATH=${PYTHONPATH}:/liaoweiduo/continual-learning-baselines
    EXPERIMENTS: 
    test case 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --exp_name Naive --cuda 0
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --exp_name ER --cuda 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --exp_name LwF --cuda 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --exp_name GEM --cuda 3
    
    test case 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --freeze --exp_name Naive --cuda 0
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --freeze --exp_name ER --cuda 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --freeze --exp_name LwF --cuda 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --freeze --exp_name GEM --cuda 3
    
    test case 3
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --exp_name Naive --cuda 0
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --exp_name ER --cuda 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --exp_name LwF --cuda 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --exp_name GEM --cuda 3
    
    test case 4
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --freeze --exp_name Naive --cuda 0
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --freeze --exp_name ER --cuda 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --freeze --exp_name LwF --cuda 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --label_map --freeze --exp_name GEM --cuda 3
    
    Non comp test case
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --non_comp --exp_name nc-Naive --cuda 0
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --non_comp --exp_name nc-ER --cuda 1
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --non_comp --exp_name nc-LwF --cuda 2
    python experiments/split_sub_vqa/naive_novel.py --use_wandb --non_comp --exp_name nc-GEM --cuda 3
    '''

