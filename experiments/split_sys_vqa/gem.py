import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import GEM
from avalanche.training.plugins import EvaluationPlugin

from models.resnet import ResNet18, MTResNet18
from models.cnn_128 import CNN128, MTCNN128
from experiments.utils import set_seed, create_default_args, create_experiment_folder
from datasets.cgqa import SplitSysGQA


def gem_ssysvqa_ti(override_args=None):
    """
    GEM algorithm on split systematic VQA on task-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 4, 'num_train_samples_each_label': 10000, 'train_mb_size': 100,
        'eval_every': 100, 'eval_mb_size': 50,
        'patterns_per_exp': 256, 'mem_strength': 0.5,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'TIME',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments'
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name if args.exp_name != "TIME" else None,
        project_name=args.project_name)
    args.exp_name = exp_path.split(os.sep)[-1]
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    benchmark = SplitSysGQA(n_experiences=args.n_experiences, return_task_id=True, seed=1234, shuffle=True,
                            dataset_root=args.dataset_root,
                            num_samples_each_label=args.num_train_samples_each_label)
    if args.model == "resnet":
        model = MTResNet18(pretrained=args.pretrained, pretrained_model_path=args.pretrained_model_path)
    elif args.model == "cnn":
        model = MTCNN128()
    else:
        raise Exception("Un-recognized model structure.")

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    loggers = [interactive_logger, text_logger]
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = avl.logging.WandBLogger(
            project_name=args.project_name, run_name=args.exp_name,
            log_artifacts=True,
            path=checkpoint_path,
            dir=exp_path,
            config=vars(args),
        )
        loggers.append(wandb_logger)

    # ####################
    # EVALUATION PLUGIN
    # ####################
    evaluation_plugin = EvaluationPlugin(
        metrics.accuracy_metrics(minibatch=True, stream=True),
        metrics.loss_metrics(minibatch=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes,
                                         save_image=True if args.use_wandb else False,
                                         stream=True),
        benchmark=benchmark,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    cl_strategy = GEM(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
        train_mb_size=args.train_mb_size,
        train_epochs=1,
        eval_mb_size=args.eval_mb_size,
        device=device,
        evaluator=evaluation_plugin,
        eval_every=args.eval_every,
        peval_mode="iteration",
    )

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience, [benchmark.test_stream[experience.current_experience]])   # only eval self
        # cl_strategy.train(experience, benchmark.test_stream)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))

    # ####################
    # STORE CHECKPOINT
    # ####################
    # if wandb_logger is not None:
    #     wandb_logger: avl.logging.WandBLogger
    #     wandb_logger.log_artifacts
    model_file = os.path.join(checkpoint_path, 'model.pth')
    print("Store checkpoint in", model_file)
    torch.save(model.state_dict(), model_file)
    if wandb_logger is not None:

        wandb_logger: avl.logging.WandBLogger

        artifact = wandb_logger.wandb.Artifact('WeightCheckpoint', type="model")
        artifact_name = os.path.join("Models", 'WeightCheckpoint.pth')
        artifact.add_file(model_file, name=artifact_name)
        wandb_logger.wandb.run.log_artifact(artifact)

    print("Final results:")
    print(results)

    return results


def gem_ssysvqa_ci(override_args=None):
    """
    GEM algorithm on split systematic VQA on class-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 10, 'epochs': 50, 'train_mb_size': 32,
        'eval_every': 10, 'eval_mb_size': 50,
        'patterns_per_exp': 32, 'mem_strength': 0.5,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'TIME',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments',
        'return_test': True,
        'interactive_logger': True,
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name if args.exp_name != "TIME" else None,
        project_name=args.project_name)
    args.exp_name = exp_path.split(os.sep)[-1]
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    benchmark = SplitSysGQA(n_experiences=args.n_experiences, return_task_id=False, seed=1234, shuffle=True,
                            return_test=args.return_test,
                            dataset_root=args.dataset_root)
    if args.model == "resnet":
        model = ResNet18(pretrained=args.pretrained, pretrained_model_path=args.pretrained_model_path)
    # elif args.model == "cnn":
    #     model = CNN128()
    else:
        raise Exception("Un-recognized model structure.")

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, f'log_{args.exp_name}.txt'), 'a'))
    if args.interactive_logger:
        loggers = [interactive_logger, text_logger]
    else:
        loggers = [text_logger]
    wandb_logger = None
    if args.use_wandb:
        wandb_logger = avl.logging.WandBLogger(
            project_name=args.project_name, run_name=args.exp_name,
            log_artifacts=True,
            path=checkpoint_path,
            dir=exp_path,
            config=vars(args),
        )
        loggers.append(wandb_logger)

    # ####################
    # EVALUATION PLUGIN
    # ####################
    evaluation_plugin = EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes,
                                         save_image=True if args.use_wandb else False,
                                         stream=True),
        benchmark=benchmark,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    cl_strategy = GEM(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=args.eval_mb_size,
        device=device,
        evaluator=evaluation_plugin,
        # eval_every=args.eval_every,
        # peval_mode="epoch",
    )

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes: ", [
            benchmark.original_map_int_label_to_tuple[cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
        ])
        cl_strategy.train(experience,
                          num_workers=8, pin_memory=False)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream[:experience.current_experience+1],
                                        num_workers=8, pin_memory=False))

    # ####################
    # STORE CHECKPOINT
    # ####################
    # if wandb_logger is not None:
    #     wandb_logger: avl.logging.WandBLogger
    #     wandb_logger.log_artifacts
    model_file = os.path.join(checkpoint_path, 'model.pth')
    print("Store checkpoint in", model_file)
    torch.save(model.state_dict(), model_file)
    if wandb_logger is not None:

        wandb_logger: avl.logging.WandBLogger

        artifact = wandb_logger.wandb.Artifact(f'WeightCheckpoint-{args.exp_name}', type="model")
        artifact_name = os.path.join("Models", 'WeightCheckpoint.pth')
        artifact.add_file(model_file, name=artifact_name)
        wandb_logger.wandb.run.log_artifact(artifact)

    print("Final results:")
    print(results)

    # ####################
    # STORE RESULTS
    # ####################
    stored_results = []
    for result in results:
        re = dict()
        for key, item in result.items():
            if 'ConfusionMatrix' not in key:
                re[key] = item
        stored_results.append(re)
    np.save(os.path.join(exp_path, f'results-{args.exp_name}.npy'), stored_results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--model", type=str, default='resnet', help="In [resnet, cnn]")
    parser.add_argument("--pretrained", action='store_true', help='Whether to load pretrained resnet and in eval mode.')
    parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
    parser.add_argument("--exp_name", type=str, default='TIME')
    parser.add_argument("--setting", type=str, default='task', help="task: Task IL or class: class IL")
    args = parser.parse_args()

    if args.setting == 'task':
        res = gem_ssysvqa_ti(vars(args))
    elif args.setting == 'class':
        res = gem_ssysvqa_ci(vars(args))
    else:
        raise Exception("Unimplemented setting.")


    '''
    export PYTHONPATH=${PYTHONPATH}:/liaoweiduo/continual-learning-baselines
    EXPERIMENTS: 
    python experiments/split_sys_vqa/gem.py --use_wandb --model resnet --exp_name Resnet-GEM --cuda 0
    
    python experiments/split_sys_vqa/gem.py --use_wandb --setting class --exp_name GEM --cuda 0
    '''

