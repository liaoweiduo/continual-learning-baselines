import sys
import os
python_path = os.path.join(os.path.abspath('.').split('continual-learning-baselines')[0],
                           'continual-learning-baselines')
# python_path = '/liaoweiduo/continual-learning-baselines'
sys.path.append(python_path)
print(f'Add python path: {python_path}')

import copy
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin

from models.resnet import get_resnet
from experiments.utils import set_seed, create_default_args, create_experiment_folder, \
    get_strategy, get_benchmark, get_model
from strategies.cam import CAMPluginMetric

from experiments.config import default_args, FIXED_CLASS_ORDER


def fewshot_test(override_args=None):
    args = create_default_args(default_args, override_args)

    print(vars(args))
    assert (
        args.dataset_mode not in ['continual']
    ), f"dataset mode should not be {args.dataset_mode}."
    if args.dataset_mode in ['nonf', 'nono', 'sysf', 'syso']:
        assert (
            args.test_n_way == 2
        ), f"Few-shot tasks should be 2-way for {args.dataset_mode} mode. " \
           f"But current you specify test_n_way: {args.test_n_way}."
    assert (args.exp_name != "TIME"
            ), f"exp_name should be one of trained experiment path."
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name,
        project_name=args.project_name)
    args.exp_name = f'{args.exp_name}-{args.dataset_mode}-{args.strategy}'      # Naive-tsk-sys-naive
    if args.test_freeze_feature_extractor:
        args.exp_name = f'{args.exp_name}-frz'
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    if not args.ignore_finished_testing:
        '''Check whether this testing has been done'''
        if os.path.exists(os.path.join(exp_path, f'results-{args.exp_name}.npy')):
            print(f"{args.exp_name} has alreadly finished. Pass.")
            return None

    # ####################
    # BENCHMARK
    # ####################
    task_offset = 10 if args.return_task_id else 1
    benchmark = get_benchmark(args, task_offset=task_offset)

    # ####################
    # MODEL
    # ####################
    origin_model = get_model(args, checkpoint_path=checkpoint_path)

    # ####################
    # LOGGER
    # ####################
    loggers = []
    if args.use_text_logger:
        loggers.append(avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a')))
    if args.use_interactive_logger:
        loggers.append(avl.logging.InteractiveLogger())

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
    metrics_list = [
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
    ]
    if args.use_cam_visualization:
        assert (-1 < args.test_task_id < args.test_n_experiences
                ), f"error specify test_task_id: {args.test_task_id}"
        metrics_list.append(CAMPluginMetric(args.image_size,
                                            benchmark=benchmark,
                                            wandb_log=args.use_wandb,
                                            num_samples=5,
                                            target=args.test_task_id))

    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        # benchmark=benchmark,
        loggers=loggers)

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results, accs = [], []
    if args.test_task_id == -1:
        tasks = benchmark.train_stream
    else:
        tasks = [benchmark.train_stream[args.test_task_id]]
    for experience in tasks:
        current_experience = experience.current_experience
        print("Start of experience ", current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes: ", [
            benchmark.label_info[2][cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[current_experience]
        ])
        model = copy.deepcopy(origin_model)

        # ####################
        # STRATEGY INSTANCE
        # ####################
        cl_strategy = get_strategy(args.strategy, model, benchmark, device, evaluation_plugin, args, early_stop=False)

        cl_strategy.train(experience, pin_memory=False, num_workers=10)
        print("Training completed")

        print("Computing accuracy on the whole test set.")
        results.append(cl_strategy.eval(benchmark.test_stream[current_experience], pin_memory=False, num_workers=10))

        task_id_str = '%03d' % (task_offset + current_experience)    # 010, 011  -> 309
        print(f"Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}: ",
              results[-1][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])
        accs.append(results[-1][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])

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
    if args.test_task_id == -1:
        file_name = f'results-{args.exp_name}.npy'
    else:
        file_name = f'results-{args.exp_name}-{args.test_task_id}.npy'
    np.save(os.path.join(exp_path, file_name), stored_results)

    print('###################################')
    print('accs:', accs)

    avg_test_acc = np.mean(accs)
    std_test_acc = np.std(accs)
    ci95_test_acc = 1.96 * (std_test_acc/np.sqrt(args.test_n_experiences))
    print(f'Top1_Acc_Stream/eval_phase/test_stream: '
          f'{avg_test_acc*100:.2f}% +- {ci95_test_acc * 100:.2f}%)')
    if wandb_logger is not None:
        wandb_logger.wandb.log({'avg_test_acc': avg_test_acc,
                                'std_test_acc': std_test_acc,
                                'ci95_test_acc': ci95_test_acc})

    # finish wandb
    if wandb_logger is not None:
        wandb_logger.wandb.finish()

    return results


if __name__ == "__main__":

    results = fewshot_test()

    # dataset_modes = ['sys', 'pro', 'sub', 'non', 'noc']         # cgqa
    # dataset_modes = ['sys', 'pro', 'non', 'noc']         # cpin

    '''Naive'''
    # for dataset_mode in dataset_modes:
    #     fewshot_test({
    #         'use_wandb': False, 'project_name': 'CPIN', 'return_task_id': False, 'use_interactive_logger': False,
    #         'test_freeze_feature_extractor': True, 'dataset': 'cpin',
    #         'exp_name': 'Naive-cls', 'strategy': 'naive', 'dataset_mode': dataset_mode,
    #         'learning_rate': 0.01,
    #     })

    '''ER'''
    # for dataset_mode in dataset_modes:
    #     fewshot_test({
    #         'use_wandb': False, 'project_name': 'CPIN', 'return_task_id': False, 'use_interactive_logger': False,
    #         'test_freeze_feature_extractor': True, 'dataset': 'cpin',
    #         'exp_name': 'ER-cls', 'strategy': 'naive', 'dataset_mode': dataset_mode,
    #         'learning_rate': 0.01,
    #     })

    '''GEM'''
    # for dataset_mode in dataset_modes:
    #     fewshot_test({
    #         'use_wandb': False, 'project_name': 'CPIN', 'return_task_id': False, 'use_interactive_logger': False,
    #         'test_freeze_feature_extractor': True, 'dataset': 'cpin',
    #         'exp_name': 'GEM-cls', 'strategy': 'naive', 'dataset_mode': dataset_mode,
    #         'learning_rate': 0.01, 'gem_patterns_per_exp': 256, 'gem_mem_strength': 0.3,
    #     })

    '''LwF'''
    # for dataset_mode in dataset_modes:
    #     fewshot_test({
    #         'use_wandb': False, 'project_name': 'CPIN', 'return_task_id': False, 'use_interactive_logger': False,
    #         'test_freeze_feature_extractor': True, 'dataset': 'cpin',
    #         'exp_name': 'LwF-cls', 'strategy': 'naive', 'dataset_mode': dataset_mode,
    #         'learning_rate': 0.01, 'lwf_alpha': 1, 'lwf_temperature': 2,
    #     })

    '''EWC'''
    # for dataset_mode in dataset_modes:
    #     fewshot_test({
    #         'use_wandb': False, 'project_name': 'CPIN', 'return_task_id': False, 'use_interactive_logger': False,
    #         'test_freeze_feature_extractor': True, 'dataset': 'cpin',
    #         'exp_name': 'EWC-cls', 'strategy': 'naive', 'dataset_mode': dataset_mode,
    #         'learning_rate': 0.01, 'ewc_lambda': 1,
    #     })

# CUDA_VISIBLE_DEVICES=2 python experiments/fewshot_testing.py > ../avalanche-experiments/CGQA/GEM-cls/fewshot_testing-naive-frz.out 2>&1

# CUDA_VISIBLE_DEVICES=0 python experiments/fewshot_testing.py > ../avalanche-experiments/CPIN/Naive-tsk/fewshot_testing-naive-frz.out 2>&1


