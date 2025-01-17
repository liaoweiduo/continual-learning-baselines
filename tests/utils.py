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


def template_exp_sh(target, path, name, params, out_path='../avalanche-experiments/out/task.out', cuda=0):
    """
    Generate sh file from 1 params dict
    :param target: experiments/continual_training.py or experiments/fewshot_testing.py
    :param path: store the sh, 'tests/tasks/TASK_NAME'
    :param name: sh file name, '1'
    :param params: a list of param dict
    :param out_path: path to the root of std out file.  No use
    :param cuda: device used
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    template_str = \
        f"#!/bin/sh\n" \
        f"export WANDB_MODE=offline\n" \
        "abs_path=`pwd`\n"\
        "echo abs_path:$abs_path\n"\
        "export PYTHONPATH=${PYTHONPATH}:${abs_path}\n"

    for param_idx, param in enumerate(params):
        '''Param to str'''
        param_str = ''
        for key, value in param.items():
            if type(value) is not bool:
                param_str += f" --{key} '{value}'"
            elif value is True:
                param_str += f" --{key}"
        # param_str = ' '.join([f"--{key} {value}" for key, value in params.items() if type(value) is not bool])
        # param_str_ = ' '.join([f"--{key}" for key, value in params.items() if value is True])       # True
        # param_str = ' '.join([param_str, param_str_])

        # template_str += \
        #     f"CUDA_VISIBLE_DEVICES={cuda} python3 {target}{param_str}" \
        #     f" >> {out_path} 2>&1\n"
        template_str += \
            f"CUDA_VISIBLE_DEVICES={cuda} python3 {target}{param_str}\n"

    '''Write to file'''
    with open(os.path.join(path, f'{name}.sh'), 'w', newline='') as f:
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

    task_str = f"#!/bin/sh"

    '''Generate json'''
    for idx, name in enumerate(name_list):
        task_str += f"\ntaiji_client start -scfg {name}.json"

        config = {
            "Token": "bv3uQFYl4YCVLkWfEcfLsQ",
            "business_flag": "AILab_MLC_CQ",
            "start_cmd": f"sh {cmd_path}/{name}.sh",
            "model_local_file_path": "/apdcephfs/private_yunqiaoyang/private_weiduoliao/continual-learning-baselines/",
            "host_num": 1,
            "host_gpu_num": 1,
            "GPUName": "V100",
            "is_elasticity": True,
            "mount_ceph_business_flag": "DrugAI_CQ",
            "image_full_name": "mirrors.tencent.com/yunqiao_cv/liaoweiduo:avalanche-0.3.1"
        }
        with open(os.path.join(path, f'{name}.json'), 'w') as f:
            json.dump(config, f, indent=4)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w', newline='') as f:
        f.write(task_str)


def template_sustech(name_list, cmd_path, path):
    """
    Generate slurm bash for file_list and 1 sh contains all sbatch $run_id$.slurm
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    task_str = f"#!/bin/sh"

    '''Generate slurm bash'''
    for idx, name in enumerate(name_list):
        task_str += f"\nsh slurm{name}.sh"

        template_str = \
            f"#!/bin/bash\n" \
            f"cd ~\n" \
            f"sbatch avalanche_bash.slurm {name} {path.split('/')[-1]}\n"

        '''Write to file'''
        with open(os.path.join(path, f'slurm{name}.sh'), 'w', newline='') as f:
            f.write(template_str)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w', newline='') as f:
        f.write(task_str)


def template_hisao(name_list, cmd_path, path):
    """
    Generate bash for file_list and 1 sh contains all sh $run_id$.bash.
    this bash is to cd the working path.
    """
    '''Make dir'''
    if not os.path.exists(path):
        os.makedirs(path)

    task_str = f"#!/bin/sh"

    '''Generate slurm bash'''
    for idx, name in enumerate(name_list):
        task_str += f"\nsh {name}.bash >> {name}.out 2>&1"

        template_str = \
            f"#!/bin/sh\n" \
            f"cd ../../../continual-learning-baselines\n" \
            f"sh {cmd_path}/{name}.sh\n"

        '''Write to file'''
        with open(os.path.join(path, f'{name}.bash'), 'w', newline='') as f:
            f.write(template_str)

    '''Generate task.sh'''
    with open(os.path.join(path, 'task.sh'), 'w', newline='') as f:
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
