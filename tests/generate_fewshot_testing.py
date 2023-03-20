import sys
import os
python_path = os.path.join(os.path.abspath('.').split('continual-learning-baselines')[0],
                           'continual-learning-baselines')
# python_path = '/liaoweiduo/continual-learning-baselines'
sys.path.append(python_path)
print(f'Add python path: {python_path}')


import time
import wandb
import copy

from experiments.continual_training import continual_train
from tests.utils import template_exp_sh, template_tencent, return_time


def generate_params(common_args, param_grid):
    keys = set(param_grid.keys())

    print(param_grid)

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
        param.update(common_args)

    return params


def main(params):
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
    iter = 0
    for idx, param in enumerate(params):
        if len(params_temp) < num_runs_1sh:
            params_temp.append(param)

        if len(params_temp) == num_runs_1sh or idx == len(params) - 1:  # every num_runs_1sh or last runs
            print(f'Generate {iter}.sh with params: {params_temp}.')
            template_exp_sh(
                target='experiments/fewshot_testing.py',
                path=f'../avalanche-experiments/tasks/{task_name}',
                name=iter,
                params=params_temp,
                # out_path=f"{exp_root}/out/{task_name}-{iter}.out",
            )
            names.append(iter)
            params_temp = []
            iter += 1

    '''Generate json and sh for Tencent servers'''
    template_tencent(
        name_list=names,
        cmd_path=f'{task_root}/{task_name}',
        path=f'../avalanche-experiments/tasks/{task_name}')


task_name = return_time()   # defined by time
task_root = 'tests/tasks'        # path for sh file from code_root
num_runs_1sh = 2       # num of runs in 1 sh file
common_args = {
    'model_backbone': 'resnet18',
    'use_wandb': False,
    'use_interactive_logger': False,
    'project_name': 'CGQA',
    'dataset': 'cgqa',
    'dataset_root': '/apdcephfs/share_1364275/lwd/datasets',
    'exp_root': '/apdcephfs/share_1364275/lwd/avalanche-experiments',
    'learning_rate': 0.001,
    'test_freeze_feature_extractor': True,
}

params = []

"""
exp: fresh or old concepts
"""
# param_grid = {
#     'train_class_order': ['fixed'],
#     'test_n_way': [2],
#     'dataset_mode': ['nonf', 'nono', 'sysf', 'syso'],
#     'exp_name': ['naive-cls-lr0_003', 'naive-tsk-lr0_008',
#                  'er-cls-lr0_003', 'er-tsk-lr0_0008',
#                  'gem-cls-lr0_01-p32-m0_3', 'gem-tsk-lr0_001-p32-m0_3',
#                  'lwf-cls-lr0_005-a1-t1', 'lwf-tsk-lr0_01-a1-t1',
#                  'ewc-cls-lr0_005-lambda0_1', 'ewc-tsk-lr0_005-lambda2'],
# }
# params.extend(generate_params(common_args, param_grid))


"""
exp: different training size
"""
param_grid = {
    'exp_name': [f'train_size-{strategy}-{c_t}-ts{ts}'
                 for strategy in ['naive', 'er', 'gem', 'lwf', 'ewc']
                 for c_t in ['cls', 'tsk']
                 for ts in [50, 200, 300, 800]],
    'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc'],
}
# c_t: [10, 100, 500, 1000]
# c_t: [50, 200, 300, 800]
params.extend(generate_params(common_args, param_grid))


"""
exp: baselines resnet cgqa
"""
# # naive
# common_args.update({
#     'exp_name': 'naive-cls-lr0_003'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# common_args.update({
#     'exp_name': 'naive-tsk-lr0_008'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# # er
# common_args.update({
#     'exp_name': 'er-cls-lr0_003'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# common_args.update({
#     'exp_name': 'er-tsk-lr0_0008'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# # gem
# common_args.update({
#     'exp_name': 'gem-cls-lr0_01-p32-m0_3'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# common_args.update({
#     'exp_name': 'gem-tsk-lr0_001-p32-m0_3'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# # lwf
# common_args.update({
#     'exp_name': 'lwf-cls-lr0_005-a1-t1'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# common_args.update({
#     'exp_name': 'lwf-tsk-lr0_01-a1-t1'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# # ewc
# common_args.update({
#     'exp_name': 'ewc-cls-lr0_005-lambda0_1'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))
#
# common_args.update({
#     'exp_name': 'ewc-tsk-lr0_005-lambda2'
# })
# param_grid = {
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc']
# }
# params.extend(generate_params(common_args, param_grid))


main(params)
