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
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage

from experiments.utils import create_default_args, create_experiment_folder, get_strategy, get_benchmark, get_model
from experiments.config import default_args, FIXED_CLASS_ORDER
from experiments.fewshot_testing import fewshot_test
from tests.utils import get_average_metric
from strategies.select_module import SelectionPluginMetric, ImageSimilarityPluginMetric


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
    RNGManager.set_random_seeds(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK
    # ####################
    benchmark = get_benchmark(args)

    # ####################
    # CHECKPOINTING
    # ####################
    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(directory=checkpoint_path),
        map_location=device
    )
    # Load checkpoint (if exists in the given storage)
    # If it does not exist, strategy will be None and initial_exp will be 0
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    wandb_logger = None
    # image_similarity_plugin_metric = None
    selection_plugin_metric = None
    if strategy is None:
        # ####################
        # MODEL
        # ####################
        model = get_model(args)

        # ####################
        # LOGGER
        # ####################
        loggers = []
        if args.use_text_logger:
            loggers.append(avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a')))
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
        if args.strategy == 'our':
            # image_similarity_plugin_metric = ImageSimilarityPluginMetric(wandb_log=False, image_size=args.image_size)
            selection_plugin_metric = SelectionPluginMetric(benchmark, sparse_threshold=args.ssc_threshold)
        metrics_list = [
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
            metrics.forgetting_metrics(experience=True, stream=True),
            metrics.class_accuracy_metrics(stream=True),
            metrics.timing_metrics(epoch=True),
            metrics.disk_usage_metrics(paths_to_monitor=exp_path, epoch=True),      # only train , experience=True
            metrics.cpu_usage_metrics(epoch=True),            # only train , experience=True
            metrics.ram_usage_metrics(epoch=True),                # only train , experience=True
        ]
        if args.cuda >= 0:
            metrics_list.append(metrics.gpu_usage_metrics(args.cuda, epoch=True))      # only train , experience=True
        if args.strategy == 'our':
            metrics_list.extend([
                selection_plugin_metric,
                # image_similarity_plugin_metric,
            ])
        # if args.dataset_mode == 'continual':
        #     metrics_list.extend([
        #         metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes,
        #                                          save_image=True if args.use_wandb else False,
        #                                          stream=True),
        #     ])
        evaluation_plugin = EvaluationPlugin(
            *metrics_list,
            # benchmark=benchmark,
            loggers=loggers)

        # ####################
        # STRATEGY INSTANCE
        # ####################
        strategy = get_strategy(args.strategy, model, benchmark, device, evaluation_plugin, args,
                                early_stop=not args.disable_early_stop, plugins=[checkpoint_plugin])
    else:
        model = strategy.model
        # if args.strategy == 'our':
        #     image_similarity_plugin_metric = [
        #         metric for metric in strategy.evaluator.metrics if isinstance(metric, ImageSimilarityPluginMetric)
        #     ][0]        # will raise exception if ImageSimilarityPluginMetric not in strategy.evaluator.metrics


    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    num_trained_exp_this_run = 0
    results = []
    for exp_idx, (experience, val_task) in enumerate(
            zip(benchmark.train_stream, benchmark.val_stream)):
        if exp_idx < initial_exp:
            continue    # when initial_exp == len(train_stream), ValueError raises for train_stream[initial_exp:]

        if 0 <= args.train_num_exp <= initial_exp + num_trained_exp_this_run:
            break       # initial_exp is num of exps have been done before the script.

        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        if hasattr(benchmark, 'label_info'):
            print("Current Classes: ", [
                benchmark.label_info[2][cls_idx]
                for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
            ])

        # if experience.current_experience == 0:    # issue: will has two /Task000/Exp000
        #     print("Testing random model")
        #     image_similarity_plugin_metric.set_active(True)
        #     results.append(strategy.eval(benchmark.test_stream))
        #     image_similarity_plugin_metric.set_active(False)

        strategy.train(experience, eval_streams=[val_task], pin_memory=False, num_workers=10)
        print("Training completed")

        print("Computing accuracy on the whole test set.")

        # if image_similarity_plugin_metric is not None:
        #     image_similarity_plugin_metric.set_active(True)

        result = strategy.eval(benchmark.test_stream, pin_memory=False, num_workers=10)
        results.append(result)

        # if image_similarity_plugin_metric is not None:
        #     image_similarity_plugin_metric.set_active(False)

        # ####################
        # STORE CHECKPOINT
        # ####################
        # if args.use_wandb:
        #
        #     wandb_logger: avl.logging.WandBLogger
        #
        #     artifact = wandb_logger.wandb.Artifact(f'WeightCheckpoint-{args.exp_name}', type="model")
        #     artifact_name = os.path.join("Models", 'WeightCheckpoint.pth')
        #     artifact.add_file(model_file, name=artifact_name)
        #     wandb_logger.wandb.run.log_artifact(artifact)
        if not args.do_not_store_checkpoint_per_exp:
            model_file = os.path.join(checkpoint_path, f'model-{experience.current_experience}.pth')
            print("Store checkpoint in", model_file)
            torch.save(model.state_dict(), model_file)
        model_file = os.path.join(checkpoint_path, 'model.pth')
        print("Store checkpoint in", model_file)
        torch.save(model.state_dict(), model_file)

        # ####################
        # STORE RESULTS
        # ####################
        stored_results = []

        re = dict()
        for key, item in result.items():
            if 'ConfusionMatrix' not in key:
                re[key] = item
        stored_results.append(re)

        result_file = os.path.join(exp_path, f'results-{args.exp_name}-{experience.current_experience}.npy')
        print("Save results in", result_file)
        np.save(result_file, stored_results)

        num_trained_exp_this_run += 1

    # print("Final results:")
    # print(results)

    print("Experiments completed")

    # if num_trained_exp_this_run == 0:       # no training performed, only test
    #     print("No training is performed, just computing accuracy on the whole test set.")
    #
    #     result = strategy.eval(benchmark.test_stream, pin_memory=False, num_workers=10)
    #     results.append(result)
    #     stored_results = []
    #     re = dict()
    #     for key, item in result.items():
    #         if 'ConfusionMatrix' not in key:
    #             re[key] = item
    #     stored_results.append(re)
    #     result_file = os.path.join(exp_path, f'results-{args.exp_name}-only-test.npy')
    #     print("Save results in", result_file)
    #     np.save(result_file, stored_results)
    ## RuntimeError: Checkpoint file /liaoweiduo/avalanche-experiments/CGQA/
    ## Multi_Label-concept-tsk_True-lr0_0001/Checkpoints/10/checkpoint.pth already exists.

    if len(results) > 0:
        '''print needed info'''
        avg_test_acc = get_average_metric(results[-1], 'Top1_Acc_Stream/eval_phase/test_stream')
        print('Average test acc:', avg_test_acc)
        if wandb_logger is not None:
            wandb_logger.wandb.log({'avg_test_acc': avg_test_acc})
        # average test accuracy for all tasks

    # finish wandb
    if wandb_logger is not None:
        wandb_logger.wandb.finish()

    return results


if __name__ == "__main__":

    # results = continual_train({
    #     'use_wandb': False, 'project_name': 'CGQA',
    #     'return_task_id': False, 'use_interactive_logger': True,
    #     'dataset': 'cgqa',
    #     'exp_name': 'Naive-cls', 'strategy': 'naive',
    #     'learning_rate': 0.001,
    # })

    results = continual_train()
    # CUDA_VISIBLE_DEVICES=4 python experiments/continual_training.py

    '''fewshot test'''
    if default_args['do_fewshot_testing']:
        common_args = {
            'use_wandb': False,
            'use_interactive_logger': False,
            'use_text_logger': True,
            'learning_rate': 0.001,
            'epochs': 20,
            'disable_early_stop': True,
            'eval_every': -1,
            'test_freeze_feature_extractor': True,
            'strategy': default_args['strategy'] if default_args['strategy'] in ['our'] else 'naive',
        }
        # exp_name should be fixed.
        if default_args['dataset'] == 'cobj':
            fewshot_sets = ['sys', 'pro', 'non', 'noc']
        else:
            fewshot_sets = ['sys', 'pro', 'sub', 'non', 'noc']
        for dataset_mode in fewshot_sets:
            common_args['dataset_mode'] = dataset_mode
            fewshot_test(common_args)
