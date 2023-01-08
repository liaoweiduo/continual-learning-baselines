import sys
sys.path.append('/liaoweiduo/continual-learning-baselines')

import time
import wandb
import copy

from experiments.continual_training import continual_train

return_task_id = False
target = 'ewc'      # optional: [naive, er, gem, lwf, ewc]

# param name should be consistent with key.
common_args = {
    'use_wandb': True,
    'use_interactive_logger': False,
    'return_task_id': return_task_id,
    'strategy': target,
}
if target == 'naive':
    exp_name_template = 'Naive-' + \
                        ('tsk' if return_task_id else 'cls') + \
                        '-lr{learning_rate}'
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    }
elif target == 'er':
    exp_name_template = 'ER-' + \
                        ('tsk' if return_task_id else 'cls') + \
                        '-lr{learning_rate}'
    param_grid = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    }
elif target == 'gem':
    exp_name_template = 'GEM-' + \
                        ('tsk' if return_task_id else 'cls') + \
                        '-lr{learning_rate}-p{gem_patterns_per_exp}-m{gem_mem_strength}'
    param_grid = {
        'learning_rate': [0.0005, 0.001],
        'gem_patterns_per_exp': [128, 256],
        'gem_mem_strength': [0.3, 0.5],
    }
elif target == 'lwf':
    exp_name_template = 'LwF-' + \
                        ('tsk' if return_task_id else 'cls') + \
                        '-lr{learning_rate}-a{lwf_alpha}-t{lwf_temperature}'
    param_grid = {
        'learning_rate': [0.005, 0.01],
        'lwf_alpha': [1, 5, 10],
        'lwf_temperature': [1, 2],
    }
elif target == 'ewc':
    exp_name_template = 'EWC-' + \
                        ('tsk' if return_task_id else 'cls') + \
                        '-lr{learning_rate}-lambda{ewc_lambda}'
    param_grid = {
        'learning_rate': [0.005, 0.01, 0.05],
        'ewc_lambda': [0.5, 1, 1.5],
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
    param.update(common_args)
    exp_name = exp_name_template.format(**param)
    exp_name = exp_name.replace('.', '_')      # for float, change '0.1' to '0_1' for potential problem in Windows.
    param['exp_name'] = exp_name
    print(f'{time.asctime(time.localtime(time.time()))}: Run experiment with params: {param}.')

    res = continual_train(param)

    wandb.finish()

print(f'{time.asctime(time.localtime(time.time()))}: Complete tuning hyper parameters for {exp_name_template}.')
print('************************')
print()
# CMD:
# CUDA_VISIBLE_DEVICES=0 python tests/hyper_param_tune.py > ../avalanche-experiments/out/hyper_param_naive_cls.out 2>&1
# CUDA_VISIBLE_DEVICES=1 python tests/hyper_param_tune.py > ../avalanche-experiments/out/hyper_param_ewc_tsk.out 2>&1
# >> for append
