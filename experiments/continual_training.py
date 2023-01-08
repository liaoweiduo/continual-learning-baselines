import sys
# sys.path.append('.')
sys.path.append('/liaoweiduo/continual-learning-baselines')

import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin

from models.resnet import get_resnet
from experiments.utils import set_seed, create_default_args, create_experiment_folder, get_strategy
from datasets.cgqa import continual_training_benchmark

from experiments.config import default_args, FIXED_CLASS_ORDER


def continual_train(override_args=None):
    args = create_default_args(default_args, override_args)
    print(vars(args))
    assert (args.dataset_mode == 'continual'
            ), f"dataset mode is {args.dataset_mode}, need to be `continual`."
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name if args.exp_name != "TIME" else None,
        project_name=args.project_name)
    args.exp_name = exp_path.split(os.sep)[-1]
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    shuffle = True if args.train_class_order == 'shuffle' else False
    fixed_class_order = None if shuffle else FIXED_CLASS_ORDER[args.dataset_mode]
    benchmark = continual_training_benchmark(
        n_experiences=args.n_experiences, return_task_id=args.return_task_id,
        seed=args.seed, fixed_class_order=fixed_class_order, shuffle=shuffle,
        dataset_root=args.dataset_root)
    if args.model_backbone == "resnet18":
        model = get_resnet(
            multi_head=args.return_task_id,
            pretrained=args.model_pretrained, pretrained_model_path=args.pretrained_model_path)
    else:
        raise Exception(f"Un-recognized model structure {args.model_backbone}.")

    # ####################
    # LOGGER
    # ####################
    loggers = [
        avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    ]
    if args.use_interactive_logger:
        loggers.append(avl.logging.InteractiveLogger())
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
    metrics_list = [
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
    ]
    if args.dataset_mode == 'continual':
        metrics_list.extend([
            metrics.forgetting_metrics(experience=True, stream=True),
            metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes,
                                             save_image=True if args.use_wandb else False,
                                             stream=True),
        ])
    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        benchmark=benchmark,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    cl_strategy = get_strategy(args.strategy, model, device, evaluation_plugin, args, early_stop=True)

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience, val_task in zip(benchmark.train_stream, benchmark.val_stream):
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes: ", [
            benchmark.label_info[2][cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
        ])
        cl_strategy.train(experience, eval_streams=[val_task])
        print("Training completed")

        print("Computing accuracy on the whole test set.")
        results.append(cl_strategy.eval(benchmark.test_stream))

    # ####################
    # STORE CHECKPOINT
    # ####################
    # if wandb_logger is not None:
    #     wandb_logger: avl.logging.WandBLogger
    #     wandb_logger.log_artifacts
    model_file = os.path.join(checkpoint_path, 'model.pth')
    print("Store checkpoint in", model_file)
    torch.save(model.state_dict(), model_file)
    if args.use_wandb:

        wandb_logger: avl.logging.WandBLogger

        artifact = wandb_logger.wandb.Artifact(f'WeightCheckpoint-{args.exp_name}', type="model")
        artifact_name = os.path.join("Models", 'WeightCheckpoint.pth')
        artifact.add_file(model_file, name=artifact_name)
        wandb_logger.wandb.run.log_artifact(artifact)

    print("Final results:")
    print(results)

    # ####################
    # STORE RESULTS
    # ####################
    stored_results = []
    for result in results:
        re = dict()
        for key, item in result.items():
            if 'ConfusionMatrix' not in key:
                re[key] = item
        stored_results.append(re)
    np.save(os.path.join(exp_path, f'results-{args.exp_name}.npy'), stored_results)

    return results


if __name__ == "__main__":

    '''Naive'''
    # results = continual_train({
    #     'use_wandb': False, 'return_task_id': False, 'use_interactive_logger': True,
    #     'exp_name': 'Naive-cls', 'strategy': 'naive',
    #     'learning_rate': 0.001,
    # })

    '''ER'''
    # results = continual_train({
    #     'use_wandb': False, 'return_task_id': False, 'use_interactive_logger': True,
    #     'exp_name': 'ER-cls', 'strategy': 'er',
    #     'learning_rate': 0.001,
    # })

    '''GEM'''
    results = continual_train({
        'use_wandb': False, 'return_task_id': True, 'use_interactive_logger': True,
        'exp_name': 'GEM-tsk', 'strategy': 'gem',
        'learning_rate': 0.005, 'gem_patterns_per_exp': 256, 'gem_mem_strength': 0.3,
    })

    '''LwF'''
    # results = continual_train({
    #     'use_wandb': False, 'return_task_id': False, 'use_interactive_logger': True,
    #     'exp_name': 'LwF-cls', 'strategy': 'lwf',
    #     'learning_rate': 0.01, 'lwf_alpha': 1, 'lwf_temperature': 2,
    # })

    '''EWC'''
    # results = continual_train({
    #     'use_wandb': False, 'return_task_id': False, 'use_interactive_logger': True,
    #     'exp_name': 'EWC-cls', 'strategy': 'ewc',
    #     'learning_rate': 0.005, 'ewc_lambda': 1,
    # })

    # CUDA_VISIBLE_DEVICES=0 python experiments/continual_training.py
