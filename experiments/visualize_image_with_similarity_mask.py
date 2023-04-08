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

import wandb

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.evaluation.metrics.images_samples import ImagesSamplePlugin

from experiments.utils import create_default_args, create_experiment_folder, get_strategy
from experiments.config import default_args, FIXED_CLASS_ORDER
from experiments.fewshot_testing import fewshot_test
from tests.utils import get_average_metric
from strategies.select_module import SelectionPluginMetric, ImageSimilarityPluginMetric


def main(override_args=None):
    args = create_default_args(default_args, override_args)
    print(vars(args))
    assert (args.dataset_mode == 'continual'
            ), f"dataset mode is {args.dataset_mode}, need to be `continual`."
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name if args.exp_name != "TIME" else None,
        project_name=args.project_name)
    args.exp_name = exp_path.split(os.sep)[-1]
    RNGManager.set_random_seeds(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    shuffle = True if args.train_class_order == 'shuffle' else False
    fixed_class_order = None if shuffle else FIXED_CLASS_ORDER[args.dataset_mode]
    if args.dataset == 'cgqa':
        from datasets.cgqa import continual_training_benchmark
    elif args.dataset == 'cpin':
        from datasets.cpin import continual_training_benchmark
    else:
        raise Exception(f'Un-implemented dataset: {args.dataset}.')
    if args.model_backbone == 'vit':
        from datasets.cgqa import build_transform_for_vit

        train_transform = build_transform_for_vit((args.image_size, args.image_size), True)
        eval_transform = build_transform_for_vit((args.image_size, args.image_size), False)
    else:
        train_transform, eval_transform = None, None    # default transform
    benchmark = continual_training_benchmark(
        n_experiences=args.n_experiences, image_size=(args.image_size, args.image_size),
        return_task_id=args.return_task_id,
        seed=args.seed, fixed_class_order=fixed_class_order, shuffle=shuffle,
        dataset_root=args.dataset_root,
        train_transform=train_transform, eval_transform=eval_transform,
        num_samples_each_label=args.num_samples_each_label
    )

    # ####################
    # CHECKPOINTING
    # ####################
    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(directory=checkpoint_path),
        map_location=device
    )
    # Load checkpoint (if exists in the given storage)
    # If it does not exist, strategy will be None and initial_exp will be 0

    # import pathlib
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath

    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    # pathlib.PosixPath = temp

    wandb_logger = None
    if strategy is None:
        assert args.strategy == 'our'
        from models.module_net import get_module_net
        model = get_module_net(
            args=args,
            multi_head=args.return_task_id,
            # pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth')
        )
    else:
        model = strategy.model
        wandb.finish()

    model = model.to(device)

    loggers = []
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

    image_similarity_plugin_metric = ImageSimilarityPluginMetric(wandb_log=True)
    # image_sample_plugin = ImagesSamplePlugin(mode='eval', n_cols=5, n_rows=4)
    metrics_list = [
        image_similarity_plugin_metric,
        SelectionPluginMetric(),
        # image_sample_plugin,
    ]
    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    strategy = get_strategy(args.strategy, model, benchmark, device, evaluation_plugin, args,
                            early_stop=not args.disable_early_stop, plugins=[])

    print("No training is performed, just run plugin.")
    results = []
    image_similarity_plugin_metric.set_active(True)
    results.append(strategy.eval(benchmark.test_stream[0]))
    image_similarity_plugin_metric.set_active(False)

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
    result_file = os.path.join(exp_path, f'results-{args.exp_name}-image-with-similarity-mask.npy')
    print("Save results in", result_file)
    np.save(result_file, stored_results)

    # finish wandb
    if wandb_logger is not None:
        wandb_logger.wandb.finish()

    return results


if __name__ == "__main__":

    results = main()
