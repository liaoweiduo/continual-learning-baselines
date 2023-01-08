import os
from datetime import datetime

from types import SimpleNamespace
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random

from avalanche.training.plugins import EarlyStoppingPlugin


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


def get_strategy(name, model, device, evaluator, args, early_stop=True):
    if early_stop:
        plugins = [EarlyStoppingPlugin(patience=args.eval_patience, val_stream_name='val_stream')]
        eval_every = args.eval_every
    else:
        plugins = None
        eval_every = -1

    if name == 'naive':
        from avalanche.training.supervised import Naive
        return Naive(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
    )
    elif name == 'gem':
        from avalanche.training.supervised import GEM
        return GEM(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            patterns_per_exp=args.gem_patterns_per_exp, memory_strength=args.gem_mem_strength,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'lwf':
        from avalanche.training.supervised import LwF
        return LwF(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            alpha=args.lwf_alpha, temperature=args.lwf_temperature,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'er':
        from avalanche.training.supervised import Replay
        return Replay(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            mem_size=args.er_mem_size,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'ewc':
        from avalanche.training.supervised import EWC
        return EWC(
            model,
            torch.optim.Adam(model.parameters(), lr=args.learning_rate),
            CrossEntropyLoss(),
            ewc_lambda=args.ewc_lambda,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    else:
        raise Exception(f"Un-implemented strategy: {name}.")
