from pathlib import Path
import inspect
from pandas import read_csv
import os
from datetime import datetime
import json

import tests


def pandas_to_list(input_str):
    return [float(el) for el in input_str.strip('[] ').split(' ')]


def get_target_result(strat_name: str, bench_name: str):
    """
    Read the target_results.csv file and retrieve the target performance for
    the given strategy on the given benchmark.
    :param strat_name: strategy name as found in the target file
    :param bench_name: benchmark name as found in the target file
    :return: target performance (either a float or a list of floats)
    """

    p = os.path.join(Path(inspect.getabsfile(tests)).parent, 'target_results.csv')
    data = read_csv(p, sep=',', comment='#')
    target = data[(data['strategy'] == strat_name) & (data['benchmark'] == bench_name)]['result'].values[0]
    if isinstance(target, str) and target.startswith('[') and target.endswith(']'):
        target = pandas_to_list(target)
    else:
        target = float(target)
    return target


def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
    """
    Compute the average of a metric based on the provided metric name.
    The average is computed across the instance of the metrics containing the
    given metric name in the input dictionary.
    :param metric_dict: dictionary containing metric name as keys and metric value as value.
        This dictionary is usually returned by the `eval` method of Avalanche strategies.
    :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
    :return: a number representing the average of all the metric containing `metric_name` in their name
    """

    avg_stream_acc = []
    for k, v in metric_dict.items():
        if k.startswith(metric_name):
            avg_stream_acc.append(v)
    return sum(avg_stream_acc) / float(len(avg_stream_acc))


def return_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def template_exp_sh(target, path, name, params, out_path='../avalanche-experiments', cuda=0):
    """
    Generate sh file from 1 params dict
    :param target: experiments/continual_training.py or experiments/fewshot_testing.py
    :param path: store the sh, 'tests/tasks/TASK_NAME'
    :param name: sh file name, '1'
    :param params: a dict of params
    :param out_path: path to the root of std out file.
    :param cuda: device used
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    '''Params to str'''
    param_str = ' '.join([f'--{key} {value}' for key, value in params.items() if type(value) is not bool]) + ' '
    param_str += ' '.join([f'--{key}' for key, value in params.items() if value is True])       # True

    template_str = \
        f"#!/bin/bash \n" \
        f"export WANDB_MODE=offline \n" \
        f"CUDA_VISIBLE_DEVICES={cuda} python {target} {param_str} \\\n" \
        f"> {out_path}/{params['project_name']}/{params['exp_name']}/{params['exp_name']}.out 2>&1"

    '''Write to file'''
    with open(os.path.join(path, f'{name}.sh'), 'w') as f:
        f.write(template_str)


def template_tencent(name_list, cmd_path, path):
    """
    Generate jsons for file_list and 1 sh contains all taiji_client start -scfg config.json
    :param name_list: a list of file: [0, 1, 2, ...]
    :param cmd_path: path start from project_root to the sh file
    :param path:  store the sh and json, 'tests/tasks/TASK_NAME'
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    task_str = f"#!/bin/bash"

    '''Generate json'''
    for idx, name in enumerate(name_list):
        task_str += f"\ntaiji_client start -scfg {name}.json"

        config = {
            "Token": "bv3uQFYl4YCVLkWfEcfLsQ",
            "business_flag": "AILab_MLC_CQ",
            "start_cmd": f"bash {cmd_path}/{name}.sh",
            "model_local_file_path": "/apdcephfs/private_yunqiaoyang/private_weiduoliao/continual-learning-baselines/",
            "host_num": 1,
            "host_gpu_num": 1,
            "GPUName": "V100",
            "is_elasticity": True,
            "mount_ceph_business_flag": "DrugAI_CQ",
            "image_full_name": "mirrors.tencent.com/yunqiao_cv/liaoweiduo:avalanche0"
        }
        with open(os.path.join(path, f'{name}.json'), 'w') as f:
            json.dump(config, f, indent=4)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w') as f:
        f.write(task_str)


if __name__ == '__main__':
    # template_exp_sh('experiments/continual_training.py',
    #                 f'tasks/{return_time()}',
    #                 '1',
    #                 {'a': True, 'b': False, 'c': 1, 'd': 1.0, 'project_name': 'CGQA', 'exp_name': 'exp'})

    template_tencent(
        [0, 1, 2],
        f'tests/tasks',
        f'tasks/{return_time()}'
    )
