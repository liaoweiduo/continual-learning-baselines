import os
from datetime import datetime

from types import SimpleNamespace
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random

from avalanche.training.plugins import EarlyStoppingPlugin, LRSchedulerPlugin


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


def get_strategy(name, model, benchmark, device, evaluator, args, early_stop=True, plugins=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def make_scheduler(_optimizer):
        if args.lr_schedule == 'step':
            _scheduler = torch.optim.lr_scheduler.StepLR(
                _optimizer, step_size=args.lr_schedule_step_size, gamma=args.lr_schedule_gamma
            )
        elif args.lr_schedule == 'cos':
            _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, args.epochs, args.lr_schedule_eta_min)
        else:
            raise Exception(f'Un-implemented lr schedule: {args.lr_schedule}.')
        return _scheduler

    if plugins is None:
        plugins = []
    if args.lr_schedule != 'none':
        plugins.append(LRSchedulerPlugin(scheduler=make_scheduler(optimizer)))

    eval_every = args.eval_every
    if early_stop:
        plugins.append(EarlyStoppingPlugin(patience=args.eval_patience, val_stream_name='val_stream'))

    if name == 'naive':
        from avalanche.training import Naive
        return Naive(
            model,
            optimizer,
            CrossEntropyLoss(),
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'gem':
        from avalanche.training import GEM
        return GEM(
            model,
            optimizer,
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
        from avalanche.training import LwF
        return LwF(
            model,
            optimizer,
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
        from avalanche.training import Replay
        return Replay(
            model,
            optimizer,
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
        from avalanche.training import EWC
        return EWC(
            model,
            optimizer,
            CrossEntropyLoss(),
            ewc_lambda=args.ewc_lambda,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'our':
        from strategies.select_module import Algorithm
        return Algorithm(
            model,
            optimizer,
            CrossEntropyLoss(),
            benchmark=benchmark,
            ssc=args.ssc, ssc_threshold=args.ssc_threshold,
            scc=args.scc,
            isc=args.isc, csc=args.csc,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    elif name == 'concept':
        from avalanche.training import Naive
        from strategies.multi_concept_classifier import MultiConceptClassifier

        plugins.append(MultiConceptClassifier(model, benchmark, weight=args.multi_concept_weight))

        return Naive(
            model,
            optimizer,
            CrossEntropyLoss(),
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every, peval_mode="epoch",
        )
    else:
        raise Exception(f"Un-implemented strategy: {name}.")
