import os
from datetime import datetime

from types import SimpleNamespace
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import random

import avalanche
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


def get_model(args, checkpoint_path=None, checkpoint_model_id=-1, multi_task_baseline=False):
    # '''Check resume'''
    # if os.path.exists(os.path.join(checkpoint_path, 'model.pth')):
    #     pretrained, pretrained_model_path = True, os.path.join(checkpoint_path, 'model.pth')
    # else:
    #     pretrained, pretrained_model_path = args.model_pretrained, args.pretrained_model_path
    if args.dataset_mode in ['continual', 'sysfull', 'profull', 'subfull', 'nonfull', 'nocfull']:   # treat these as con
        pretrained, pretrained_model_path = args.model_pretrained, args.pretrained_model_path
        multi_head = args.return_task_id
        fix = False
    elif args.test_on_random_model:
        pretrained, pretrained_model_path = False, args.pretrained_model_path
        multi_head = True
        fix = args.test_freeze_feature_extractor
    else:
        assert checkpoint_path is not None
        if checkpoint_model_id == -1:
            model_name = 'model.pth'
        else:
            model_name = f'model-{checkpoint_model_id}.pth'
        pretrained, pretrained_model_path = True, os.path.join(checkpoint_path, model_name)
        multi_head = True
        fix = args.test_freeze_feature_extractor

    if args.strategy == 'our':
        from models.module_net import get_module_net
        model = get_module_net(
            args=vars(args),
            multi_head=multi_head,
            pretrained=pretrained, pretrained_model_path=pretrained_model_path,
            masking=True if not multi_task_baseline else False,
            fix=fix)
    elif args.model_backbone == "resnet18":
        from models.resnet import get_resnet
        model = get_resnet(
            multi_head=multi_head,
            initial_out_features=100 if args.strategy == 'icarl' else 2,
            pretrained=pretrained, pretrained_model_path=pretrained_model_path,
            masking=True if not multi_task_baseline else False,
            add_multi_class_classifier=True if args.strategy == 'concept' else False,
            fix=fix,
            normal_classifier=True if args.strategy == 'icarl' else False,
        )
        if args.strategy == 'icarl':
            from avalanche.models import initialize_icarl_net
            model.apply(initialize_icarl_net)
    elif args.model_backbone == "vit":
        from models.vit import get_vit
        model = get_vit(
            image_size=args.image_size,
            multi_head=multi_head,
            pretrained=pretrained, pretrained_model_path=pretrained_model_path,
            fix=fix,
            masking=True if not multi_task_baseline else False,
            patch_size=args.vit_patch_size, dim=args.vit_dim, depth=args.vit_depth, heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim, dropout=args.vit_dropout, emb_dropout=args.vit_emb_dropout)
    else:
        raise Exception(f"Un-recognized model structure {args.model_backbone}.")

    return model


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
        # val_stream_name = 'train_stream' if args.dataset == 'scifar100' else 'val_stream'
        val_stream_name = 'val_stream'
        plugins.append(EarlyStoppingPlugin(patience=args.eval_patience, val_stream_name=val_stream_name))

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
        from avalanche.training import Naive
        from avalanche.training.plugins import ReplayPlugin

        plugins.append(ReplayPlugin(args.er_mem_size, task_balanced_dataloader=True))

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
    elif name == 'icarl':
        from avalanche.training import ICaRL
        from datasets.cgqa import _build_default_transform
        return ICaRL(
            model.resnet if hasattr(model, 'resnet') else model.vit,
            model.classifier,
            optimizer,
            args.icarl_mem_size,
            buffer_transform=_build_default_transform((args.image_size, args.image_size)),
            fixed_memory=True,
            train_mb_size=args.train_mb_size,
            train_epochs=args.epochs,
            eval_mb_size=args.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every,
        )
    elif name == 'agem':
        from avalanche.training import AGEM
        return AGEM(
            model,
            optimizer,
            CrossEntropyLoss(),
            patterns_per_exp=args.agem_patterns_per_exp, sample_size=args.agem_sample_size,
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

        plugins.append(MultiConceptClassifier(model, benchmark,
                                              weight=args.multi_concept_weight,
                                              mask_origin_loss=args.mask_origin_loss))

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


def get_benchmark(args, task_offset=0, multi_task=False):
    from experiments.config import FIXED_CLASS_ORDER

    shuffle = True if args.train_class_order == 'shuffle' else False
    fixed_class_order = None if shuffle else FIXED_CLASS_ORDER[args.dataset_mode]

    if args.dataset_mode == 'continual':

        if args.dataset == 'cgqa':
            from datasets.cgqa import continual_training_benchmark
        elif args.dataset == 'cpin':
            from datasets.cpin import continual_training_benchmark
        elif args.dataset == 'cobj':
            from datasets.cobj import continual_training_benchmark
        elif args.dataset == 'scifar100':
            from datasets.scifar100 import continual_training_benchmark
        else:
            raise Exception(f'Un-implemented dataset: {args.dataset}.')

        if args.model_backbone == 'vit':
            from datasets.cgqa import build_transform_for_vit

            train_transform = build_transform_for_vit((args.image_size, args.image_size), True)
            eval_transform = build_transform_for_vit((args.image_size, args.image_size), False)
        else:
            train_transform, eval_transform = None, None    # default transform

        # if args.dataset == 'scifar100':
        #     if train_transform is None:     # if set as None, it will not use default setting.
        #         benchmark = avalanche.benchmarks.SplitCIFAR100(
        #             args.n_experiences, return_task_id=args.return_task_id,
        #             dataset_root=args.dataset_root
        #         )
        #     else:
        #         benchmark = avalanche.benchmarks.SplitCIFAR100(
        #             args.n_experiences, return_task_id=args.return_task_id,
        #             dataset_root=args.dataset_root,
        #             train_transform=train_transform, eval_transform=eval_transform,
        #         )
        #     benchmark.val_stream = benchmark.train_stream
        # else:
        benchmark = continual_training_benchmark(
            n_experiences=args.n_experiences, image_size=(args.image_size, args.image_size),
            return_task_id=args.return_task_id,
            seed=args.seed, fixed_class_order=fixed_class_order, shuffle=shuffle,
            dataset_root=args.dataset_root,
            train_transform=train_transform, eval_transform=eval_transform,
            num_samples_each_label=args.num_samples_each_label,
            multi_task=multi_task
        )

    else:
        if args.dataset == 'cgqa':
            from datasets.cgqa import fewshot_testing_benchmark
        elif args.dataset == 'cpin':
            from datasets.cpin import fewshot_testing_benchmark
        elif args.dataset == 'cobj':
            from datasets.cobj import fewshot_testing_benchmark
        else:
            raise Exception(f'Un-implemented dataset: {args.dataset}.')

        benchmark = fewshot_testing_benchmark(
            n_experiences=args.test_n_experiences, image_size=(args.image_size, args.image_size),
            mode=args.dataset_mode,
            n_way=args.test_n_way, n_shot=args.test_n_shot, n_val=args.test_n_val, n_query=args.test_n_query,
            task_offset=task_offset,
            seed=args.seed, fixed_class_order=fixed_class_order,
            dataset_root=args.dataset_root)

    return benchmark
