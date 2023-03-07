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
from experiments.utils import set_seed, create_default_args, create_experiment_folder, get_strategy

from experiments.config import default_args, FIXED_CLASS_ORDER


def fewshot_test(override_args=None):
    args = create_default_args(default_args, override_args)

    print(vars(args))
    assert (args.dataset_mode in ['sys', 'pro', 'sub', 'non', 'noc']
            ), f"dataset mode is {args.dataset_mode}, need to be one of ['sys', 'pro', 'sub', 'non', 'noc']."
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

    # ####################
    # BENCHMARK & MODEL
    # ####################
    shuffle = True if args.train_class_order == 'shuffle' else False
    fixed_class_order = None if shuffle else FIXED_CLASS_ORDER[args.dataset_mode]
    task_offset = 10 if args.return_task_id else 1
    if args.dataset == 'cgqa':
        from datasets.cgqa import fewshot_testing_benchmark
    elif args.dataset == 'cpin':
        from datasets.cpin import fewshot_testing_benchmark
    else:
        raise Exception(f'Un-implemented dataset: {args.dataset}.')
    benchmark = fewshot_testing_benchmark(
        n_experiences=args.test_n_experiences, image_size=(args.image_size, args.image_size), mode=args.dataset_mode,
        n_way=args.test_n_way, n_shot=args.test_n_shot, n_val=args.test_n_val, n_query=args.test_n_query,
        task_offset=task_offset,
        seed=args.seed, fixed_class_order=fixed_class_order,
        dataset_root=args.dataset_root)
    if args.model_backbone == "resnet18":
        origin_model = get_resnet(
            multi_head=True,
            pretrained=True, pretrained_model_path=os.path.join(checkpoint_path, 'model.pth'),
            fix=args.test_freeze_feature_extractor,
        )
    elif args.model_backbone == "vit":
        from models.vit import get_vit
        origin_model = get_vit(
            image_size=args.image_size,
            multi_head=True,
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

    # ####################
    # EVALUATION PLUGIN
    # ####################
    metrics_list = [
            metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
            metrics.loss_metrics(epoch=True, experience=True, stream=True),
    ]
    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        # benchmark=benchmark,
        loggers=loggers)

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results, accs = [], []
    for experience, val_task in zip(benchmark.train_stream, benchmark.val_stream):
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
        cl_strategy = get_strategy(args.strategy, model, device, evaluation_plugin, args, early_stop=True)

        cl_strategy.train(experience, eval_streams=[val_task], pin_memory=False, num_workers=10)
        print("Training completed")

        print("Computing accuracy on the whole test set.")
        results.append(cl_strategy.eval(benchmark.test_stream[current_experience], pin_memory=False, num_workers=10))

        task_id_str = '%03d' % (task_offset + current_experience)    # 010, 011  -> 309
        print(f"Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}: ",
              results[current_experience][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])
        accs.append(results[current_experience][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])

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

    print('###################################')
    print('accs:', accs)
    print(f'Top1_Acc_Stream/eval_phase/test_stream: '
          f'{np.mean(accs)*100:.2f}% +- {1.96 * (np.std(accs)/np.sqrt(args.test_n_experiences)) * 100:.2f}%)')

    # finish wandb
    if args.use_wandb:
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


