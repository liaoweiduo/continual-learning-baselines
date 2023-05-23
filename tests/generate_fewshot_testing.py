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
from tests.utils import template_exp_sh, template_hisao, template_sustech, return_time


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
        path=f'../avalanche-experiments/tasks/{task_name}')


target = 'experiments/fewshot_testing.py'
task_name = return_time()   # defined by time
print(task_name)
# task_root = 'tests/tasks'        # path for sh in the working path
task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
num_runs_1sh = 6       # num of runs in 1 sh file
fix_device = False      # cuda self-increase for each run if True, else use cuda:0
start_iter = 0
common_args = {
    'use_wandb': False,
    'use_interactive_logger': True,
    'project_name': 'CGQA',
    'dataset': 'cgqa',
    # 'dataset_root': '/apdcephfs/share_1364275/lwd/datasets',
    # 'exp_root': '/apdcephfs/share_1364275/lwd/avalanche-experiments',
    'learning_rate': 0.001,
    'epochs': 20,       # 20
    'test_freeze_feature_extractor': True,
    'disable_early_stop': True,
    'eval_every': -1,   # do not enable eval during training
}

params = []

"""
exp: cobj baselines
"""
# common_args.update({
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'project_name': 'COBJ',
#     'dataset': 'cobj',
#     'test_n_way': 6,        # [3, 6, 10]
# })
# param_grid = {
#     'exp_name': [         # 3-tasks 10-way
#         'HT-MT-3tasks-naive-tsk_True-lr0_001', 'HT-MT-3tasks-naive-tsk_False-lr0_001',
#         'HT-naive-tsk_True-lr0_001', 'HT-naive-tsk_False-lr0_001',
#         'HT-er-tsk_True-lr0_01', 'HT-er-tsk_False-lr0_01',
#         'HT-gem-tsk_True-lr0_001-p16-m0_3', 'HT-gem-tsk_False-lr0_001-p256-m0_00139',
#         'HT-lwf-tsk_True-lr0_001-a1-t2', 'HT-lwf-tsk_False-lr0_001-a1-t1_52',
#         'HT-ewc-tsk_True-lr0_01-lambda100', 'HT-ewc-tsk_False-lr0_00053-lambda10',
#     ],
#     # 'exp_name': [           # 10-tasks 3-way
#     #     'HT-MT-naive-tsk_True-lr0_00231', 'HT-MT-naive-tsk_False-lr0_00123',
#     #     'HT-10tasks-naive-tsk_True-lr1e-05', 'HT-10tasks-naive-tsk_False-lr0_001',
#     #     'HT-10tasks-er-tsk_True-lr0_001', 'HT-10tasks-er-tsk_False-lr0_001',
#     #     'HT-10tasks-gem-tsk_True-lr0_001-p16-m0_3', 'HT-10tasks-gem-tsk_False-lr0_001-p256-m0_00139',
#     #     'HT-10tasks-lwf-tsk_True-lr1e-05-a1-t2', 'HT-10tasks-lwf-tsk_False-lr0_001-a1-t2',
#     #     'HT-10tasks-ewc-tsk_True-lr1e-05-lambda100', 'HT-10tasks-ewc-tsk_False-lr0_01-lambda10',
#     # ],
#     # 'exp_name': [           # 5-tasks 6-way
#     #     'HT-MT-5tasks-naive-tsk_True-lr0_001', 'HT-MT-5tasks-naive-tsk_False-lr0_001',
#     #     'HT-5tasks-naive-tsk_True-lr0_01', 'HT-5tasks-naive-tsk_False-lr0_001',
#     #     'HT-5tasks-er-tsk_True-lr0_001', 'HT-5tasks-er-tsk_False-lr0_001',
#     #     'HT-5tasks-gem-tsk_True-lr0_01-p16-m0_3', 'HT-5tasks-gem-tsk_False-lr0_01-p256-m0_00139',
#     #     'HT-5tasks-lwf-tsk_True-lr0_001-a1-t2', 'HT-5tasks-lwf-tsk_False-lr0_001-a1-t2',
#     #     'HT-5tasks-ewc-tsk_True-lr0_001-lambda100', 'HT-5tasks-ewc-tsk_False-lr0_001-lambda10',
#     # ],
#     'dataset_mode': ['sys', 'pro', 'non', 'noc'],
# }
# params.extend(generate_params(common_args, param_grid))


"""
exp: do fewshot testing on random model
"""
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# common_args.update({
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'test_on_random_model': True,
# })
# param_grid = {
#     # 'exp_name': [
#     #     f'concept-concept-tsk_{return_task_id}-lr{learning_rate}-w{multi_concept_weight}'
#     #     for learning_rate in ['0_0001', '0_001', '0_01', '0_1']
#     #     for multi_concept_weight in ['0_5', '1', '2']
#     #     for return_task_id in [True, False]
#     # ],
#     'exp_name': [
#         f'random-naive-tsk_False',
#     ],
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc'],
# }
# params.extend(generate_params(common_args, param_grid))

"""
exp: assist with multi-concept learning head
"""
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# common_args.update({
#     'use_interactive_logger': True,
# })
# param_grid = {
#     # 'exp_name': [
#     #     f'concept-concept-tsk_{return_task_id}-lr{learning_rate}-w{multi_concept_weight}'
#     #     for learning_rate in ['0_0001', '0_001', '0_01', '0_1']
#     #     for multi_concept_weight in ['0_5', '1', '2']
#     #     for return_task_id in [True, False]
#     # ],
#     'exp_name': [
#         f'concept-concept-tsk_True-lr0_01-w1',
#         f'concept-concept-tsk_False-lr0_001-w0_5',
#     ],
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc'],
# }
# params.extend(generate_params(common_args, param_grid))


"""
exp: baselines vit cgqa
"""
# task_root = 'tests/tasks'        # path for sh in the working path
# # task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# num_runs_1sh = 5       # num of runs in 1 sh file
# fix_device = True      # cuda self-increase for each run if True, else use cuda:0
# common_args.update({
#     'use_interactive_logger': False,
#     'use_text_logger': True,
#     'model_backbone': 'vit',
#     'image_size': 224,
#     'train_mb_size': 32,
# })
# param_grid = {
#     'exp_name': [
#         # 'HT-MT-vit-naive-tsk_False-lr0_0001',
#         'HT-MT-vit-naive-tsk_True-lr0_0001',
#         # 'ht-vit-naive-cls-lr0_0001', 'ht-vit-naive-tsk-lr0_0001',
#         # 'ht-vit-er-cls-lr0_0001', 'ht-vit-er-tsk-lr0_0001',
#         # 'ht-vit-gem-cls-lr5e-05', 'ht-vit-gem-tsk-lr1e-05',
#         # 'ht-vit-lwf-cls-lr0_0001', 'ht-vit-lwf-tsk-lr0_0001',
#         # 'ht-vit-ewc-cls-lr0_0001', 'ht-vit-ewc-tsk-lr0_0001',
#     ],
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc'],
# }
# params.extend(generate_params(common_args, param_grid))


"""
exp: fresh or old concepts
"""
# task_root = '../avalanche-experiments/tasks'        # path for sh out of working path
# fix_device = False
# common_args.update({
#     'use_interactive_logger': False,
#     'use_text_logger': True,
# })
# param_grid = {
#     'train_class_order': ['fixed'],
#     'test_n_way': [2],
#     'dataset_mode': ['nonf', 'nono', 'sysf', 'syso'],
#     'exp_name': [
#         # 'naive-cls-lr0_003', 'naive-tsk-lr0_008',
#         # 'er-cls-lr0_003', 'er-tsk-lr0_0008',
#         # 'gem-cls-lr0_01-p32-m0_3', 'gem-tsk-lr0_001-p32-m0_3',
#         # 'lwf-cls-lr0_005-a1-t1', 'lwf-tsk-lr0_01-a1-t1',
#         # 'ewc-cls-lr0_005-lambda0_1', 'ewc-tsk-lr0_005-lambda2',
#         'MT-naive-tsk_False-lr0_005', 'MT-naive-tsk_True-lr0_001',
#     ],
# }
# params.extend(generate_params(common_args, param_grid))


"""
exp: different training size
"""
# param_grid = {
#     'exp_name': [f'train_size-{strategy}-{c_t}-ts{ts}'
#                  for strategy in ['naive', 'er', 'gem', 'lwf', 'ewc']
#                  for c_t in ['cls', 'tsk']
#                  for ts in [50, 200, 300, 800]],
#     'dataset_mode': ['sys', 'pro', 'sub', 'non', 'noc'],
# }
# # c_t: [10, 100, 500, 1000]
# # c_t: [50, 200, 300, 800]
# params.extend(generate_params(common_args, param_grid))


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


main(params, fix_device, start_iter)
