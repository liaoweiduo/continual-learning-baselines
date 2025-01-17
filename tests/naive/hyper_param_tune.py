import sys
import time

import wandb

sys.path.append('.')
# sys.path.append('/liaoweiduo/continual-learning-baselines')
import copy

from experiments.split_sys_vqa.naive import naive_ssysvqa_ci
from experiments.split_sub_vqa.naive import naive_ssubvqa_ci

TBD
return_task_id=True,


target = 'sub_color'      # optional: [sys, sub_color, sub]




# param name should be consistent with key.
if target in ['sys']:
    exp_name_template = 'Naive-lr{learning_rate}'

    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    }
    common_args = {
        'return_test': False,
        'use_wandb': False,
        'cuda': 0,  # controlled use CUDA_VISIBLE_DEVICES=0
        'interactive_logger': False,
    }
elif target in ['sub']:
    exp_name_template = 'Naive-lr{learning_rate}'

    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    }
    common_args = {
        'return_test': False,
        'use_wandb': False,
        'cuda': 0,  # controlled use CUDA_VISIBLE_DEVICES=0
        'interactive_logger': False,
    }
elif target in ['sub_color']:
    exp_name_template = 'color-Naive-lr{learning_rate}'

    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    }
    common_args = {
        'return_test': False,
        'use_wandb': False,
        'cuda': 0,  # controlled use CUDA_VISIBLE_DEVICES=0
        'color_attri': True,
        'interactive_logger': False,
    }
else:
    raise Exception("Not within options!")
keys = set(param_grid.keys())


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

'''Run experiments in sequence'''
print('************************')
print(f'{time.asctime(time.localtime(time.time()))}: Start tuning hyper parameters for {exp_name_template}.')
print(param_grid)
for param in params:
    '''Generate exp_name according to param'''
    exp_name = exp_name_template.format(**param)
    exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
    param['exp_name'] = exp_name
    print(f'{time.asctime(time.localtime(time.time()))}: Run experiment with params: {param}.')
    param.update(common_args)

    if target in ['sys']:
        res = naive_ssysvqa_ci(param)
    elif target in ['sub', 'sub_color']:
        res = naive_ssubvqa_ci(param)
    else:
        raise Exception("Not within options!")

    wandb.finish()

print(f'{time.asctime(time.localtime(time.time()))}: Complete tuning hyper parameters for {exp_name_template}.')
print('************************')
print()
# CMD:
# CUDA_VISIBLE_DEVICES=0 python tests/naive/hyper_param_tune.py
# > ../avalanche-experiments/out/hyper_param_tune_naive.out 2>&1
