import sys
import os
python_path = os.path.join(os.path.abspath('.').split('continual-learning-baselines')[0],
                           'continual-learning-baselines')
# python_path = '/liaoweiduo/continual-learning-baselines'
sys.path.append(python_path)
print(f'Add python path: {python_path}')

import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin

from experiments.utils import set_seed, create_default_args, create_experiment_folder, get_strategy
from tests.utils import get_average_metric

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
    assert args.dataset == 'scifar100'

    benchmark = avl.benchmarks.SplitCIFAR100(20, return_task_id=args.return_task_id, dataset_root=args.dataset_root)
    if args.model_backbone == "resnet18":
        from models.resnet import get_resnet
        model = get_resnet(
            multi_head=args.return_task_id,
            pretrained=args.model_pretrained, pretrained_model_path=args.pretrained_model_path)
    elif args.model_backbone == "vit":
        from models.vit import get_vit
        model = get_vit(
            image_size=args.image_size,
            multi_head=args.return_task_id,
            pretrained=args.model_pretrained, pretrained_model_path=args.pretrained_model_path,
            patch_size=args.vit_patch_size, dim=args.vit_dim, depth=args.vit_depth, heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim, dropout=args.vit_dropout, emb_dropout=args.vit_emb_dropout
        )
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

        wandb_logger.wandb.watch(model)

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
        # benchmark=benchmark,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    cl_strategy = get_strategy(args.strategy, model, benchmark, device, evaluation_plugin, args, early_stop=False)

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        cl_strategy.train(experience)
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

    # print("Final results:")
    # print(results)

    '''print needed info'''
    print('Average test acc:', get_average_metric(results[-1], 'Top1_Acc_Stream/eval_phase/test_stream'))
    # average test accuracy for all tasks

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

    # finish wandb
    if args.use_wandb:
        wandb_logger.wandb.finish()

    return results


if __name__ == "__main__":

    '''Naive: cls'''
    # results = continual_train({
    #     'use_wandb': False, 'project_name': 'CGQA',
    #     'return_task_id': False, 'use_interactive_logger': True,
    #     'exp_name': 'Naive-cls', 'strategy': 'naive',
    #     'learning_rate': 0.001,
    # })

    results = continual_train({'strategy': 'er', 'dataset': 'scifar100'})
    # CUDA_VISIBLE_DEVICES=4 python experiments/continual_training.py
