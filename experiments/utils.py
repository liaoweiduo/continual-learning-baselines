import os
from datetime import datetime

from types import SimpleNamespace
import torch
import numpy as np
import random


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


def create_experiment_folder(root='.', exp_name=None, project_name=None):
    """
    generate project folder then
    generate exp folder, with a subfolder: Checkpoints to store model params.

    if project_name is None, use timestmps as folder name.
    if exp_name is None, use timestmps as folder name.
    """
    if project_name is None:
        project_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_path = os.path.abspath(os.path.join(root, project_name, exp_name))
    if os.path.exists(exp_path):
        print(f"Exist experiment with path: {exp_path}")
    else:
        print(f"Create experiment with path: {exp_path}")
        os.makedirs(exp_path)
    if not os.path.exists(os.path.join(exp_path, "Checkpoints")):
        os.makedirs(os.path.join(exp_path, "Checkpoints"))
    checkpoint_path = os.path.join(exp_path, "Checkpoints")
    return exp_path, checkpoint_path
