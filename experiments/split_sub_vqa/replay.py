import os
import argparse

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import avalanche as avl
from avalanche.evaluation import metrics as metrics
from avalanche.training.supervised import Replay
from avalanche.training.plugins import EvaluationPlugin

from models.resnet import ResNet18, MTResNet18
from models.cnn_128 import CNN128, MTCNN128
from experiments.utils import set_seed, create_default_args, create_experiment_folder
from datasets.cgqa import SplitSubGQA


def er_ssubvqa_ci(override_args=None):
    """
    ER algorithm on split substitutivity VQA on class-IL setting.
    """
    args = create_default_args({
        'cuda': 0, 'seed': 0,
        'learning_rate': 0.01, 'n_experiences': 10, 'epochs': 50, 'train_mb_size': 32,
        'eval_every': 10, 'eval_mb_size': 50,
        'mem_size': 1000,
        'model': 'resnet', 'pretrained': False, "pretrained_model_path": "../pretrained/pretrained_resnet.pt.tar",
        'use_wandb': False, 'project_name': 'Split_Sub_VQA', 'exp_name': 'TIME',
        'dataset_root': '../datasets', 'exp_root': '../avalanche-experiments',
        'return_test': True, 'color_attri': False,
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
    benchmark = SplitSubGQA(n_experiences=args.n_experiences, return_task_id=False, seed=1234, shuffle=True,
                            return_test=args.return_test, color_attri=args.color_attri,
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
    parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
    parser.add_argument("--exp_name", type=str, default='TIME')
    parser.add_argument("--non_comp", action='store_true', help="non compositional dataset")
    args = parser.parse_args()

    res = er_ssubvqa_ci(vars(args))

    '''
    export PYTHONPATH=${PYTHONPATH}:/liaoweiduo/continual-learning-baselines
    EXPERIMENTS: 
    python experiments/split_sub_vqa/replay.py --use_wandb --exp_name ER --cuda 1
    python experiments/split_sub_vqa/replay.py --use_wandb --non_comp --exp_name nc-ER --cuda 1
    '''