import sys
sys.path.append('/liaoweiduo/continual-learning-baselines')

import time
import wandb
import copy

from experiments.continual_training import continual_train
from tests.utils import template_exp_sh, template_tencent, return_time


def generate_params(common_args, param_grid, exp_name_template):
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
        '''Generate exp_name according to param'''
        param.update(common_args)
        exp_name = exp_name_template.format(**param)
        exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
        param['exp_name'] = exp_name
        print(f'Generate sh with params: {param}.')

    return params


task_name = return_time()   # defined by time
use_wandb = False
use_interactive_logger = False
project_name = 'CGQA'
dataset = 'cgqa'
dataset_root = '/apdcephfs/share_1364275/lwd/datasets'
exp_root = '/apdcephfs/share_1364275/lwd/avalanche-experiments'
common_args = {
    'use_wandb': use_wandb,
    'use_interactive_logger': use_interactive_logger,
    'project_name': project_name,
    'dataset': dataset,
    'dataset_root': dataset_root,
    'exp_root': exp_root,
}

params = []

# naive
return_task_id = False
strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}'
param_grid = {
    'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

return_task_id = True
strategy = 'naive'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}'
param_grid = {
    'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

# er
return_task_id = False
strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}'
param_grid = {
    'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

return_task_id = True
strategy = 'er'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}'
param_grid = {
    'learning_rate': [0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

# gem
return_task_id = False
strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gem_patterns_per_exp': [32, 64, 128, 256],
    'gem_mem_strength': [0.1, 0.2, 0.3, 0.4, 0.5],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

return_task_id = True
strategy = 'gem'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'gem_patterns_per_exp': [32, 64, 128, 256],
    'gem_mem_strength': [0.1, 0.2, 0.3, 0.4, 0.5],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

# lwf
return_task_id = False
strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'lwf_alpha': [0.1, 0.5, 1, 5, 10],
    'lwf_temperature': [0.1, 0.5, 1, 2],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

return_task_id = True
strategy = 'lwf'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'lwf_alpha': [0.1, 0.5, 1, 5, 10],
    'lwf_temperature': [0.1, 0.5, 1, 2],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

# ewc
return_task_id = False
strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-lambda{ewc_lambda}'
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'ewc_lambda': [0.1, 0.5, 1, 1.5, 2],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))

return_task_id = True
strategy = 'ewc'   # optional: [naive, er, gem, lwf, ewc]
common_args.update({
    'return_task_id': return_task_id,
    'strategy': strategy,
})
exp_name_template = strategy + '-' + \
                    ('tsk' if return_task_id else 'cls') + \
                    '-lr{learning_rate}-lambda{ewc_lambda}'
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'ewc_lambda': [0.1, 0.5, 1, 1.5, 2],
}
params.extend(generate_params(common_args, param_grid, exp_name_template))


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
for iter, param in enumerate(params):
    print(f'Generate sh with params: {param}.')
    template_exp_sh(
        target='experiments/continual_training.py',
        path=f'../../avalanche-experiments/tasks/{task_name}',
        name=iter,
        params=param)
    names.append(iter)

'''Generate json and sh for Tencent servers'''
template_tencent(
    name_list=names,
    cmd_path=f'../tasks/{task_name}',
    path=f'../../avalanche-experiments/tasks/{task_name}')
