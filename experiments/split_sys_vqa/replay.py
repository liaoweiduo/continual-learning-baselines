import os

import argparse
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import Naive
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin

from models.resnet import ResNet18, MTResNet18
from experiments.utils import set_seed, create_default_args, create_experiment_folder
from datasets.cgqa import SplitSysGQA


def er_ssysvqa(override_args=None):
    """
    Naive ER algorithm on split systematic VQA on task-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'train_mb_size': 100, 'epochs': 20, 'eval_mb_size': 100, 'n_experiences': 4,
        'novel_comb_epochs': 1, 'novel_comb_shot': 100,
        'mem_size': 1000,
        'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'ER-test',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments'
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(root=args.exp_root, exp_name=args.exp_name)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # ####################
    # BENCHMARK & MODEL
    # ####################
    benchmark = SplitSysGQA(n_experiences=args.n_experiences, return_task_id=True, seed=1234, shuffle=True,
                            dataset_root=args.dataset_root)
    model = MTResNet18(pretrained=args.pretrained, pretrained_model_path=args.pretrained_model_path)

    # ####################
    # LOGGER
    # ####################
    interactive_logger = avl.logging.InteractiveLogger()
    text_logger = avl.logging.TextLogger(open(os.path.join(exp_path, 'log.txt'), 'a'))
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
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        metrics.confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=True,
                                         stream=True),
        benchmark=benchmark,
        loggers=loggers)

    # ####################
    # STRATEGY INSTANCE
    # ####################
    cl_strategy = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=args.eval_mb_size,
        device=device,
        plugins=[ReplayPlugin(mem_size=args.mem_size)],
        evaluator=evaluation_plugin,
    )

    # ####################
    # TRAINING LOOP
    # ####################
    print("Starting experiment...")
    results = []
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
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

    print("Final results:")
    print(results)

    # ####################
    # NOVEL EVALUATION
    # ####################
    num_samples_each_label = args.novel_comb_shot
    '''Task id is 4'''
    benchmark_novel = SplitSysGQA(n_experiences=1, return_task_id=True, seed=1234, shuffle=True,
                                  novel_combination=True, num_samples_each_label=num_samples_each_label,
                                  task_id=args.n_experiences,   # 4
                                  dataset_root=args.dataset_root)

    evaluation_plugin_novel = EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True),
        metrics.forgetting_metrics(experience=True, stream=True),
        metrics.confusion_matrix_metrics(num_classes=benchmark_novel.n_classes, save_image=True,
                                         stream=True),
        benchmark=benchmark_novel,
        loggers=loggers)

    '''Use naive strategy to train the novel comb task'''
    cl_strategy_novel = Naive(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        train_mb_size=args.train_mb_size,
        train_epochs=args.novel_comb_epochs,
        eval_mb_size=args.eval_mb_size,
        device=device,
        plugins=[],
        evaluator=evaluation_plugin_novel,
    )
    print(f"Starting experiment on novel combination task with shot {num_samples_each_label}...")
    results_novel = []
    for experience in benchmark_novel.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy_novel.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results_novel.append(cl_strategy_novel.eval(benchmark_novel.test_stream))

    print("Novel comb results:")
    print(results_novel)

    return results, results_novel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
    parser.add_argument("--exp_name", type=str, default='ER')

    args = parser.parse_args()

    res, res_novel = er_ssysvqa(vars(args))

    import wandb

    wandb.finish()
