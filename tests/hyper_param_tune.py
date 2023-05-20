import sys
import os
python_path = os.path.join(os.path.abspath('.').split('continual-learning-baselines')[0],
                           'continual-learning-baselines')
# python_path = '/liaoweiduo/continual-learning-baselines'
sys.path.append(python_path)
print(f'Add python path: {python_path}')


import numpy as np
import time
import wandb
import copy

from experiments.continual_training import continual_train
from tests.utils import template_exp_sh, template_sustech, template_hisao, return_time


def generate_params(common_args, param_grid, exp_name_template):
    keys = set(param_grid.keys())

    print(exp_name_template, param_grid)

    def unfold(_params, _param, _choice=None):
        """recursion to get all choice of params.
            _choice: (key, value)
        """
        _param = copy.deepcopy(_param)
        if _choice is not None:     # assign value
            _param[_choice[0]] = _choice[1]

        if len(_param.keys()) == len(keys):
            '''complete'''
            _params.append(_param)
        else:
            '''select 1 unsigned key and call unfold'''
            selected_key = list(keys - set(_param.keys()))[0]
            for choice in param_grid[selected_key]:
                unfold(_params, _param, _choice=(selected_key, choice))

    '''Generate instance params for grid search in param_scope'''
    params = []
    unfold(params, dict(), None)

    for iter, param in enumerate(params):
        '''Generate exp_name according to param'''
        param.update(common_args)
        exp_name = exp_name_template.format(**param)
        exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
        param['exp_name'] = exp_name

    return params


def main(params, fix_device=True, start_iter=0):
    '''Run experiments in sequence'''
    # print('************************')
    # print(f'{time.asctime(time.localtime(time.time()))}: Start tuning hyper parameters for {exp_name_template}.')
    # print(param_grid)
    # for param in params:
    #     '''Generate exp_name according to param'''
    #     param.update(common_args)
    #     exp_name = exp_name_template.format(**param)
    #     exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
    #     param['exp_name'] = exp_name
    #     print(f'{time.asctime(time.localtime(time.time()))}: Run experiment with params: {param}.')
    #
    #     res = continual_train(param)
    #
    #     wandb.finish()
    #
    # print(f'{time.asctime(time.localtime(time.time()))}: Complete tuning hyper parameters for {exp_name_template}.')
    # print('************************')
    # print()
    # CMD:
    # CUDA_VISIBLE_DEVICES=0 python tests/hyper_param_tune.py > ../avalanche-experiments/out/hyper_param_naive_cls.out 2>&1
    # CUDA_VISIBLE_DEVICES=1 python tests/hyper_param_tune.py > ../avalanche-experiments/out/hyper_param_ewc_tsk.out 2>&1
    # >> for append

    '''Or, generate sh files'''
    names = []
    params_temp = []
    iter = start_iter
    for idx, param in enumerate(params):
        if len(params_temp) < num_runs_1sh:
            params_temp.append(param)

        if len(params_temp) == num_runs_1sh or idx == len(params) - 1:  # every num_runs_1sh or last runs
            print(f'Generate {iter}.sh with params: {params_temp}.')
            template_exp_sh(
                target=target,
                path=f'../avalanche-experiments/tasks/{task_name}',
                name=iter,
                params=params_temp,
                # out_path=f"{exp_root}/out/{task_name}-{iter}.out",
                cuda=0 if fix_device else iter,
            )
            names.append(iter)
            params_temp = []
            iter += 1

    '''Generate bash for server'''
    # template_sustech, template_hisao
    template_hisao(
        name_list=names,
        cmd_path=f'{task_root}/{task_name}',
        path=f'../avalanche-experiments/tasks/{task_name}'
    )


target = 'experiments/continual_training.py'
task_name = return_time()   # defined by time
print(task_name)
# task_root = 'tests/tasks'        # path for sh in the working path
task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
num_runs_1sh = 3       # num of runs in 1 sh file
fix_device = False      # cuda self-increase for each run if False, else use cuda:0
start_iter = 0
common_args = {
    'project_name': 'CGQA',
    'dataset': 'cgqa',
    'use_interactive_logger': True,
    'use_text_logger': False,
}
# for tencent server:
# 'dataset_root': '/apdcephfs/share_1364275/lwd/datasets',
# 'exp_root': '/apdcephfs/share_1364275/lwd/avalanche-experiments',

params = []

"""
baselines resnet agem 
"""
# target = 'experiments/continual_training.py'
# task_root = '../avalanche-experiments/tasks'
# num_runs_1sh = 2
# fix_device = False
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'agem',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'return_task_id': [False, True],
#     'learning_rate': np.around(np.logspace(-4, -1, num=8), decimals=4).tolist(),   # [1e-4,...,0.1]
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

"""
baselines resnet icarl 
"""
# target = 'experiments/continual_training.py'
# task_root = '../avalanche-experiments/tasks'
# num_runs_1sh = 3
# fix_device = False
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'icarl',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'return_task_id': False
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': np.around(np.logspace(-4, 1, num=24), decimals=4).tolist(),   # [1e-4,...,10]
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

"""
baselines resnet ssifar100
"""
# target = 'experiments/continual_training.py'
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# # multi-runs
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'naive',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
#     'skip_fewshot_testing': True,
# })
# common_args.update({
#     'strategy': 'naive',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [True],
#     'learning_rate': [0.01],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [False],
#     'learning_rate': [0.003],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# common_args.update({
#     'strategy': 'er',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [True, False],
#     'learning_rate': [0.0007],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# common_args.update({
#     'strategy': 'gem',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-p{gem_patterns_per_exp}-m{gem_mem_strength}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [True],
#     'learning_rate': [0.001],
#     'gem_patterns_per_exp': [256],
#     'gem_mem_strength': [0.7],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [False],
#     'learning_rate': [0.001],
#     'gem_patterns_per_exp': [256],
#     'gem_mem_strength': [0.1],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# common_args.update({
#     'strategy': 'lwf',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-a{lwf_alpha}-t{lwf_temperature}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [True],
#     'learning_rate': [0.001],
#     'lwf_alpha': [1],
#     'lwf_temperature': [2],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [False],
#     'learning_rate': [0.001],
#     'lwf_alpha': [1],
#     'lwf_temperature': [3.32],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# common_args.update({
#     'strategy': 'ewc',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-lambda{ewc_lambda}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [True],
#     'learning_rate': [0.01],
#     'ewc_lambda': [1],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'return_task_id': [False],
#     'learning_rate': [0.001],
#     'ewc_lambda': [0.22],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # multi-task
# target = 'experiments/multi_task_training.py'
# common_args.update({
#     'tag': 'HT-MT',
#     'strategy': 'naive',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-seed{seed}'
# param_grid = {
#     'seed': [1, 2, 3, 4, 5, 6, 7, 8],
#     'learning_rate': [0.001],
#     'return_task_id': [True, False],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

# # naive
# # param_grid = {
# #     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
# #     'return_task_id': [False, True],
# # }
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'naive',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [0.025, 0.05, 0.075, 0.1],
#     'return_task_id': [True],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'learning_rate': [3e-4, 7e-4, 3e-3, 7e-3],
#     'return_task_id': [False],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # er
# # param_grid = {
# #     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
# #     'return_task_id': [False, True],
# # }
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'er',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [3e-4, 7e-4, 3e-3, 7e-3],
#     'return_task_id': [True, False],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # gem
# # param_grid = {
# #     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
# #     'return_task_id': [False, True],
# #     'gem_patterns_per_exp': [256],      # [32, 64, 128, 256],
# #     'gem_mem_strength': [0.5],          # [0.1, 0.3, 0.5, 1.0],
# # }
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'gem',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-p{gem_patterns_per_exp}-m{gem_mem_strength}'
# param_grid = {
#     'learning_rate': [0.001],
#     'return_task_id': [False, True],
#     'gem_patterns_per_exp': [256],
#     'gem_mem_strength': [0.1, 0.3, 0.7, 1.0],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # lwf
# # param_grid = {
# #     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
# #     'return_task_id': [False, True],
# #     'lwf_alpha': [1],             # [0.1, 0.5, 1, 5, 10],
# #     'lwf_temperature': [2],       # [0.1, 0.5, 1, 2],
# # }
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'lwf',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
#     'skip_fewshot_testing': True,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-a{lwf_alpha}-t{lwf_temperature}'
# # param_grid = {
# #     'learning_rate': [0.001],
# #     'return_task_id': [False, True],
# #     'lwf_alpha': [0.01, 0.1, 10, 100],
# #     'lwf_temperature': [2],
# # }
# # param_grid = {
# #     'learning_rate': [0.001],
# #     'return_task_id': [False, True],
# #     'lwf_alpha': [1],
# #     'lwf_temperature': [0.1, 1, 10, 100],
# # }
# # param_grid = {
# #     'learning_rate': [0.001],
# #     'return_task_id': [False],
# #     'lwf_alpha': [1],
# #     'lwf_temperature': np.around(np.logspace(0, 1, num=24), decimals=2).tolist(),   # [1,...,10]
# # }
# param_grid = {
#     'learning_rate': np.around(np.logspace(-4, -2, num=8), decimals=5).tolist(),   # [1e-4,...,1e-2]
#     'return_task_id': [False],
#     'lwf_alpha': np.around(np.logspace(-2, 1, num=8), decimals=3).tolist(),   # [1e-2,...,10]
#     'lwf_temperature': [3.32],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # ewc
# common_args.update({
#     'tag': 'HT',
#     'strategy': 'ewc',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
#     'skip_fewshot_testing': True,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}' + \
#                     '-lambda{ewc_lambda}'
# # param_grid = {
# #     'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
# #     'return_task_id': [False, True],
# #     'ewc_lambda': [1],      # [0.1, 1, 10, 100]
# # }
# # param_grid = {
# #     'learning_rate': [1e-2],
# #     'return_task_id': [True],
# #     'ewc_lambda': [0.01, 0.1, 10, 100],
# # }
# # params_temp = generate_params(common_args, param_grid, exp_name_template)
# # params.extend(params_temp)
# # param_grid = {
# #     'learning_rate': [1e-3],
# #     'return_task_id': [False],
# #     'ewc_lambda': [0.01, 0.1, 10, 100],
# # }
# # params_temp = generate_params(common_args, param_grid, exp_name_template)
# # params.extend(params_temp)
# # param_grid = {
# #     'learning_rate': [1e-3],
# #     'return_task_id': [False],
# #     'ewc_lambda': np.around(np.logspace(-1, 1, num=24), decimals=2).tolist(),   # [1e-1,...,10]
# # }
# param_grid = {
#     'learning_rate': np.around(np.logspace(-4, -2, num=24), decimals=5).tolist(),   # [1e-4,...,1e-2]
#     'return_task_id': [False],
#     'ewc_lambda': [0.22, 1],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # multi-task
# target = 'experiments/multi_task_training.py'
# common_args.update({
#     'tag': 'HT-MT',
#     'strategy': 'naive',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'SCIFAR100',
#     'dataset': 'scifar100',
#     'image_size': 32,
#     'epochs': 100,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
#     'return_task_id': [False, True],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

"""
baselines resnet cobj
"""
# # Multi-task
# target = 'experiments/multi_task_training.py'
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# num_runs_1sh = 1
# start_iter = 0
# fix_device = False
# param_grid = {
#     'learning_rate': np.around(np.logspace(-4, -1, num=4), decimals=5).tolist(),
#     'return_task_id': [False, True],
# }
# common_args.update({
#     'tag': 'HT-MT-5tasks',
#     'strategy': 'naive',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     # 'test_n_way': 10,
#     'n_experiences': 5,    # [3, 5, 10]
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

# target = 'experiments/continual_training.py'
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# num_runs_1sh = 1
# fix_device = False
# # naive
# param_grid = {
#     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
#     'return_task_id': [False, True],
# }
# common_args.update({
#     'tag': 'HT-5tasks',
#     'strategy': 'naive',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'n_experiences': 5,    # [3, 5, 10]
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # er
# param_grid = {
#     'learning_rate': [1e-4, 0.001, 0.01, 0.1],
#     # [1e-5, 1e-4, 0.001, 0.01] np.around(np.logspace(-3, 0, num=8), decimals=3).tolist()
#     'return_task_id': [False, True],
# }
# common_args.update({
#     'tag': 'HT-5tasks',
#     'strategy': 'er',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'n_experiences': 5,    # [3, 5, 10]
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # gem
# common_args.update({
#     'tag': 'HT-5tasks',
#     'strategy': 'gem',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'n_experiences': 5,    # [3, 5, 10]
# })
# param_grid = {
#     'learning_rate': [1e-4, 0.001, 0.01, 0.1],
#     # np.around(np.logspace(-4, -2, num=12), decimals=5).tolist(),
#     'return_task_id': [False],
#     'gem_patterns_per_exp': [256],   # [16, 32, 64, 128, 256],
#     'gem_mem_strength': [0.00139],
# }
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'learning_rate': [1e-4, 0.001, 0.01, 0.1],
#     # np.around(np.logspace(-4, -2, num=12), decimals=5).tolist(),
#     'return_task_id': [True],
#     'gem_patterns_per_exp': [16],   # [16, 32, 64, 128, 256],
#     'gem_mem_strength': [0.3],
# }
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # lwf
# param_grid = {
#     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],     #
#     'return_task_id': [False, True],
#     'lwf_alpha': [1],  # [0.01, 0.1, 1, 10, 100]
#     'lwf_temperature': [2],
#     # np.around(np.logspace(-2, 1, num=12), decimals=3).tolist(),
# }
# common_args.update({
#     'tag': 'HT-5tasks',
#     'strategy': 'lwf',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'n_experiences': 5,    # [3, 5, 10]
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# # ewc
# common_args.update({
#     'tag': 'HT-5tasks',
#     'strategy': 'ewc',
#     'model_backbone': 'resnet18',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'n_experiences': 5,    # [3, 5, 10]
# })
# param_grid = {
#     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
#     # np.around(np.logspace(-3, 0, num=12), decimals=4).tolist(),
#     'return_task_id': [True],
#     'ewc_lambda': [100],   # [0.01, 0.1, 1, 10, 100]
# }
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-lambda{ewc_lambda}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
# param_grid = {
#     'learning_rate': [1e-5, 1e-4, 0.001, 0.01],
#     # np.around(np.logspace(-4, -2, num=12), decimals=5).tolist(),
#     'return_task_id': [False],
#     'ewc_lambda': [10],   # [0.01, 0.1, 1, 10, 100]
# }
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-lambda{ewc_lambda}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)


"""
exp: multi-task baselines
"""
# target = 'experiments/multi_task_training.py'
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# param_grid = {
#     # 'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
#     # 'return_task_id': [True, False],
#     'learning_rate': [0.001],
#     'return_task_id': [False],
# }
# common_args.update({
#     'tag': 'MT',
#     'strategy': 'naive',
#     'use_interactive_logger': False,
#     'use_text_logger': True,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)



"""
exp: multi-label baselines
"""
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# param_grid = {
#     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#     'return_task_id': [True, False],
# }
# common_args.update({
#     'tag': 'Multi_Label',
#     'strategy': 'concept',
#     'multi_concept_weight': 1,
#     'mask_origin_loss': True,
#     'skip_fewshot_testing': False,
#     'disable_early_stop': True,
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)



"""
exp: assist with multi-concept learning head
"""
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# param_grid = {
#     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#     'multi_concept_weight': [0.5, 1, 2],
#     'return_task_id': [True, False],
# }
# common_args.update({
#     'tag': 'concept',
#     'strategy': 'concept',
#     'skip_fewshot_testing': True,
#     # 'train_mb_size': 50,
# })
# exp_name_template = 'concept-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-w{multi_concept_weight}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# # for p in params_temp:
# #     p['scc'] = p['ssc']
# params.extend(params_temp)



"""
exp: module-net, 10 tasks, tune lr and reg coeff (sparse, supcon)
"""
# param_grid = {
#     'learning_rate': [1e-4],
#     'ssc': [0],
#     'scc': [0.00001, 0.0001, 0.001, 0.01],
# }
# common_args.update({
#     'tag': 'MNt1_tn',
#     'return_task_id': True,
#     'strategy': 'our',
#     'model_backbone': 'vit',
#     'image_size': 128,
#     'vit_depth': 4,
#     'use_wandb': True,
#     'train_num_exp': 1,
#     'skip_fewshot_testing': True,
#     'eval_every': 10,
#     'eval_patience': 50,
#     'epochs': 300,
#     'lr_schedule': 'cos',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
#                     '-tsk_{return_task_id}' + \
#                     '-r{scc}'
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# # for p in params_temp:
# #     p['scc'] = p['ssc']
# params.extend(params_temp)



"""
exp: module-net, only first task, tune lr and reg coeff (sparse, supcon)
"""
# param_grid = {
#     # 'learning_rate': [0.00001, 0.0001, 0.001, 0.01],
#     # 'ssc': [0.01, 0.1, 1, 10],
#     'learning_rate': [1e-5, 1e-4, 1e-3],
#     'ssc': [0, 0.1, 1, 10],
#     'scc': [0, 1, 10],
# }
# common_args.update({
#     'tag': 'MNt1_vit',
#     'return_task_id': True,
#     'strategy': 'our',
#     'model_backbone': 'vit',
#     'image_size': 128,
#     'vit_depth': 4,
#     # 'train_mb_size': 32,
#     'use_wandb': True,
#     'train_num_exp': 1,
#     'use_interactive_logger': True,
#     'skip_fewshot_testing': True,
#     # 'disable_early_stop': True,
#     'eval_every': 10,
#     'eval_patience': 50,
#     'epochs': 300,
#     'lr_schedule': 'cos',
# })
# exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + '-' + \
#                     '-tsk_{return_task_id}' + \
#                     '-lr{learning_rate}-ssc{ssc}-scc{scc}'
# # MNt1_lr_reg- for early try
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# # for p in params_temp:
# #     p['scc'] = p['ssc']
# #     # p['isc'] = p['ssc']
# #     # p['csc'] = p['ssc']
# params.extend(params_temp)



"""
exp: different training size
"""
# param_grid = {
#     # 'num_samples_each_label': [10, 100, 500, 1000]
#     'num_samples_each_label': [50, 200, 300, 800]
# }
# # naive
# return_task_id = False
# strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.003,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.008,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # er
# return_task_id = False
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.003,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.0008,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # gem
# return_task_id = False
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.01,
#     'gem_patterns_per_exp': 32,
#     'gem_mem_strength': 0.3,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.001,
#     'gem_patterns_per_exp': 32,
#     'gem_mem_strength': 0.3,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # lwf
# return_task_id = False
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.005,
#     'lwf_alpha': 1,
#     'lwf_temperature': 1,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.01,
#     'lwf_alpha': 1,
#     'lwf_temperature': 1,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # ewc
# return_task_id = False
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.005,
#     'ewc_lambda': 0.1,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.005,
#     'ewc_lambda': 2,
# })
# exp_name_template = 'train_size-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-ts{num_samples_each_label}'
# params.extend(generate_params(common_args, param_grid, exp_name_template))



"""
baselines vit cgqa
"""
# vit structure: ps16 - dim384 - depth9 - heads16 - mlp_dim1536
# Multi-task
target = 'experiments/multi_task_training.py'
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
task_root = 'tests/tasks'        # path for sh in the working path
num_runs_1sh = 1
fix_device = True
param_grid = {
    'learning_rate': [5e-5, 1e-4, 1e-3],  # np.around(np.logspace(-4, -1, num=4), decimals=5).tolist(),
    'return_task_id': [False, True],
}
common_args.update({
    'tag': 'HT-MT-vit',
    'strategy': 'naive',
    'use_interactive_logger': False,
    'use_text_logger': True,
    'project_name': 'CGQA',
    'dataset': 'cgqa',
    'model_backbone': 'vit',
    'epochs': 200,
    'image_size': 224,
    # 'train_mb_size': 32,
    'lr_schedule': 'cos',
})
exp_name_template = common_args['tag'] + '-' + common_args['strategy'] + \
                    '-tsk_{return_task_id}' + \
                    '-lr{learning_rate}'
params_temp = generate_params(common_args, param_grid, exp_name_template)
params.extend(params_temp)

# # naive
# return_task_id = False
# strategy = 'naive'
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
#
# return_task_id = True
# strategy = 'naive'
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)
#
# # er
# return_task_id = False
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # gem
# return_task_id = False
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'gem_patterns_per_exp': [32],
#     'gem_mem_strength': [0.3],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'gem_patterns_per_exp': [32],
#     'gem_mem_strength': [0.3],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # lwf
# return_task_id = False
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'lwf_alpha': [1],
#     'lwf_temperature': [1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'lwf_alpha': [1],
#     'lwf_temperature': [1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # ewc
# return_task_id = False
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'ewc_lambda': [0.1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'model_backbone': 'vit',
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = 'ht-{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 1e-3],
#     'ewc_lambda': [2],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))


# tune vit structure
# return_task_id = False
# strategy = 'naive'
# common_args.update({
#     'model_backbone': 'vit',
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'epochs': 200,
#     'image_size': 224,
#     'train_mb_size': 32,
#     'lr_schedule': 'cos',
# })
# exp_name_template = '{model_backbone}-aug-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-{lr_schedule}lr{learning_rate}-ps{vit_patch_size}' + \
#                     '-dim{vit_dim}-depth{vit_depth}-heads{vit_heads}'   # -md{vit_mlp_dim}
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
#     'vit_patch_size': [16],
#     'vit_dim': [192, 384],  # 768 is ViT-B/16
#     'vit_depth': [9],
#     'vit_heads': [16],
#     # 'vit_mlp_dim': [512, 1024],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# for p in params_temp:
#     p['vit_mlp_dim'] = 4 * p['vit_dim']
# params.extend(params_temp)
#
# param_grid = {
#     'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
#     'vit_patch_size': [16],
#     'vit_dim': [512],
#     'vit_depth': [9],
#     'vit_heads': [16],
#     'vit_mlp_dim': [512],
# }
# params_temp = generate_params(common_args, param_grid, exp_name_template)
# params.extend(params_temp)

# return_task_id = False
# strategy = 'naive'
# common_args.update({
#     'model_backbone': 'vit',
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.00001,
#     'vit_patch_size': 16,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-ps{vit_patch_size}' + \
#                     '-dim{vit_dim}-depth{vit_depth}-heads{vit_heads}-md{vit_mlp_dim}'
# param_grid = {
#     'vit_dim': [512, 1024, 2048],
#     'vit_depth': [5, 7, 9],
#     'vit_heads': [8, 16, 32],
#     'vit_mlp_dim': [512, 1024, 2048]
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = False
# strategy = 'naive'
# common_args.update({
#     'model_backbone': 'vit',
#     'return_task_id': return_task_id,
#     'strategy': strategy,
#     'learning_rate': 0.00001,
#     'vit_patch_size': 32,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-ps{vit_patch_size}' + \
#                     '-dim{vit_dim}-depth{vit_depth}-heads{vit_heads}-md{vit_mlp_dim}'
# param_grid = {
#     'vit_dim': [512, 1024, 2048],
#     'vit_depth': [5, 7, 9],
#     'vit_heads': [8, 16, 32],
#     'vit_mlp_dim': [512, 1024, 2048]
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))


"""
baselines resnet
"""
# # naive
# return_task_id = False
# strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],       # cgqa resnet
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # er
# return_task_id = False
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}'
# param_grid = {
#     'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))

# # gem
# return_task_id = False
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
# param_grid = {
#     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#     'gem_patterns_per_exp': [32, 64, 128, 256],
#     'gem_mem_strength': [0.1, 0.2, 0.3, 0.4, 0.5],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))

# return_task_id = True
# strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
# param_grid = {
#     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#     'gem_patterns_per_exp': [32, 64, 128, 256],
#     'gem_mem_strength': [0.1, 0.2, 0.3, 0.4, 0.5],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))

# # lwf
# return_task_id = False
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
# param_grid = {
#     'learning_rate': [0.001, 0.005, 0.01, 0.05],
#     'lwf_alpha': [0.1, 0.5, 1, 5, 10],
#     'lwf_temperature': [0.1, 0.5, 1, 2],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))

# return_task_id = True
# strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
# param_grid = {
#     'learning_rate': [0.001, 0.005, 0.01, 0.05],
#     'lwf_alpha': [0.1, 0.5, 1, 5, 10],
#     'lwf_temperature': [0.1, 0.5, 1, 2],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# # ewc
# return_task_id = False
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-lambda{ewc_lambda}'
# param_grid = {
#     'learning_rate': [0.001, 0.005, 0.01, 0.05],
#     'ewc_lambda': [0.1, 0.5, 1, 1.5, 2],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))
#
# return_task_id = True
# strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
# common_args.update({
#     'return_task_id': return_task_id,
#     'strategy': strategy,
# })
# exp_name_template = '{model_backbone}-' + strategy + '-' + \
#                     ('tsk' if return_task_id else 'cls') + \
#                     '-lr{learning_rate}-lambda{ewc_lambda}'
# param_grid = {
#     'learning_rate': [0.001, 0.005, 0.01, 0.05],
#     'ewc_lambda': [0.1, 0.5, 1, 1.5, 2],
# }
# params.extend(generate_params(common_args, param_grid, exp_name_template))

main(params, fix_device, start_iter)
