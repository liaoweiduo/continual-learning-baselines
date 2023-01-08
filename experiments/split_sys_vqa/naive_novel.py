import os
import argparse
import copy
import sys
import time
sys.path.append('.')

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
from datasets.cgqa import SplitSysGQA


def naive_novel_ssysvqa_ti(override_args=None):
    """
    Naive algorithm on split systematic VQA (novel comb) on task-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 1, 'num_train_samples_each_label': 5000, 'train_mb_size': 100,
        'eval_every': 50, 'eval_mb_size': 50,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'Naive',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments'
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root, exp_name=args.exp_name,
        project_name=args.project_name)
    args.exp_name = f'Novel-{args.exp_name}'

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    num_samples_each_label = args.num_train_samples_each_label
    benchmark_novel = SplitSysGQA(n_experiences=args.n_experiences, return_task_id=True, seed=1234, shuffle=True,
                                  novel_combination=True, num_samples_each_label=num_samples_each_label,
                                  dataset_root=args.dataset_root)

    '''Load model in the exp_folder and freeze the feature extractor.'''
    if args.model == "resnet":
        model = MTResNet18(pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth'))
    elif args.model == "cnn":
        model = MTCNN128(pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth'))
    else:
        raise Exception("Un-recognized model structure.")

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    loggers = [interactive_logger, text_logger]
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = avl.logging.WandBLogger(
            project_name=args.project_name, run_name=args.exp_name,     # e.g., Novel-ER
            dir=exp_path,
            config=vars(args),
        )
        loggers.append(wandb_logger)

    # ####################
    # EVALUATION PLUGIN
    # ####################
    evaluation_plugin_novel = EvaluationPlugin(
        metrics.accuracy_metrics(minibatch=True, stream=True),
        metrics.loss_metrics(minibatch=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        # metrics.confusion_matrix_metrics(num_classes=benchmark_novel.n_classes,
        #                                  save_image=True if args.use_wandb else False,
        #                                  stream=True),
        benchmark=benchmark_novel,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################

    '''Use naive strategy to train the novel comb task'''
    cl_strategy_novel = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        train_mb_size=args.train_mb_size,
        train_epochs=1,
        eval_mb_size=args.eval_mb_size,
        device=device,
        evaluator=evaluation_plugin_novel,
        eval_every=args.eval_every,
        peval_mode="iteration",
    )

    # ####################
    # TRAINING LOOP
    # ####################
    print(f"Starting experiment on novel combination task with shot {num_samples_each_label}...")
    results_novel = []
    for experience in benchmark_novel.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        cl_strategy_novel.train(experience, [benchmark_novel.test_stream], num_workers=8)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results_novel.append(cl_strategy_novel.eval(benchmark_novel.test_stream, num_workers=8))

    print("Novel comb results:")
    print(results_novel)

    return results_novel


def naive_novel_ssysvqa_ci(override_args=None):
    """
    Naive algorithm on split systematic VQA (novel comb) on class-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 300, 'epochs': 20, 'train_mb_size': 10,
        'eval_mb_size': 100,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        "freeze": False,
        "mode": "novel_test", "num_ways_each_task": 2, "num_samples_each_label": 5,
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'Naive',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments',
        'interactive_logger': True,
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root, exp_name=args.exp_name,
        project_name=args.project_name)
    if args.mode == 'novel_test':
        args.exp_name = f'Novel-{args.exp_name}'
    elif args.mode == "non_novel_test":
        args.exp_name = f'Non-novel-{args.exp_name}'
    else:
        raise Exception(f'Un-implemented mode: {args.mode}.')
    if args.freeze:
        args.exp_name = f'{args.exp_name}-frz'
    print(f'exp_name: {args.exp_name}')

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK
    # ####################
    benchmark = SplitSysGQA(n_experiences=args.n_experiences, return_task_id=False, seed=1234, shuffle=True,
                            mode=args.mode, num_ways_each_task=2, num_samples_each_label=5,
                            dataset_root=args.dataset_root)

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    if args.interactive_logger:
        loggers = [interactive_logger, text_logger]
    else:
        loggers = [text_logger]
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
    # MODEL
    # ####################
    print("Load trained model.")
    if args.model == "resnet":
        origin_model = ResNet18(initial_out_features=20,
                                pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth'),
                                fix=args.freeze)
    else:
        raise Exception("Un-recognized model structure.")

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting testing experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of testing experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes: ", [
            benchmark.original_map_int_label_to_tuple[cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
        ])

        # ####################
        # MODEL
        # ####################
        model = copy.deepcopy(origin_model)

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
        )

        cl_strategy.train(experience, num_workers=8, pin_memory=False)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream[experience.current_experience],
                                        num_workers=8, pin_memory=False))
        # print(results[experience.current_experience])
        print('Top1_Acc_Stream/eval_phase/test_stream/Task000: ',
              results[experience.current_experience]['Top1_Acc_Stream/eval_phase/test_stream/Task000'])

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
    parser.add_argument("--exp_name", type=str, default='Naive-lr0_005')
    parser.add_argument("--setting", type=str, default='class', help="task: Task IL or class: class IL")
    parser.add_argument("--freeze", action='store_true', help="whether freeze feature extractor.")
    parser.add_argument("--mode", type=str, default='novel_test', help="choice: [novel_test, non_novel_test]")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs.")
    args = parser.parse_args()

    if args.setting == 'task':
        res = naive_novel_ssysvqa_ti(vars(args))
    elif args.setting == 'class':
        res = naive_novel_ssysvqa_ci(vars(args))
    else:
        raise Exception("Unimplemented setting.")


    '''
    export PYTHONPATH=${PYTHONPATH}:/liaoweiduo/continual-learning-baselines
    EXPERIMENTS: 
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model cnn --exp_name CNN-ER --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model cnn --exp_name CNN-Naive --cuda 1
    
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model resnet --exp_name Resnet_ER --cuda 2
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model resnet --exp_name Resnet-Naive --cuda 3
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model resnet --exp_name Resnet-LwF --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --model resnet --exp_name Resnet-GEM --cuda 1
    
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --exp_name Naive --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --exp_name ER --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --exp_name LwF --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --exp_name GEM --cuda 0
    
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --freeze --exp_name Naive --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --freeze --exp_name ER --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --freeze --exp_name LwF --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --use_wandb --setting class --freeze --exp_name GEM --cuda 0
    
    novel_test and non_novel_test
    python experiments/split_sys_vqa/naive_novel.py --setting class --mode novel_test --exp_name Naive-lr0_005 --cuda 0
    python experiments/split_sys_vqa/naive_novel.py --setting class --mode non_novel_test --exp_name Naive-lr0_005 --cuda 0
    '''

