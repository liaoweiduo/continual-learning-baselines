import os
import argparse

import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import Replay
from avalanche.training.plugins import ReplayPlugin, EvaluationPlugin

from models.resnet import ResNet18, MTResNet18
from models.cnn_128 import CNN128, MTCNN128
from experiments.utils import set_seed, create_default_args, create_experiment_folder
from datasets.cgqa import SplitSysGQA


def er_ssysvqa(override_args=None):
    """
    ER algorithm on split systematic VQA on task-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 4, 'num_train_samples_each_label': 10000, 'train_mb_size': 100,
        'eval_every': 100, 'eval_mb_size': 50,
        'mem_size': 1000,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sys_VQA', 'exp_name': 'TIME',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments'
    }, override_args)
    exp_path, checkpoint_path = create_experiment_folder(
        root=args.exp_root,
        exp_name=args.exp_name if args.exp_name != "TIME" else None)
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
    cl_strategy = Replay(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        CrossEntropyLoss(),
        mem_size=args.mem_size,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--model", type=str, default='resnet', help="In [resnet, cnn]")
    parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
    parser.add_argument("--exp_name", type=str, default='TIME')
    args = parser.parse_args()

    res = er_ssysvqa(vars(args))

    '''
    export PYTHONPATH=${PYTHONPATH}:/liaoweiduo/continual-learning-baselines
    EXPERIMENTS: 
    python experiments/split_sys_vqa/replay.py --use_wandb --model resnet --exp_name Resnet-ER --cuda 1
    python experiments/split_sys_vqa/replay.py --use_wandb --model cnn --exp_name CNN-ER --cuda 2
    '''
