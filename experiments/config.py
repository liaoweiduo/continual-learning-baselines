import argparse


parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--dataset', type=str, default='cgqa', choices=['cgqa', 'cpin', 'scifar100'], metavar='DATASET',
                    help='Compositional GQA dataset: cgqa, '
                         'Compositional PIN dataset: cpin, '
                         'Split cifar100: scifar100')
parser.add_argument('--dataset_mode', type=str, default='continual', metavar='DATASET_MODE',
                    choices=['continual', 'sys', 'pro', 'sub', 'non', 'noc',
                             'nonf', 'nono', 'sysf', 'syso'],
                    help='For cgqa, mode indicates the benchmark mode.')
parser.add_argument('--return_task_id', action='store_true',
                    help='Incremental setting: True for task-IL or False for class-IL.')
parser.add_argument('--image_size', type=int, default=128, metavar='IMAGE_SIZE',
                    help='Image height and width.')
parser.add_argument('--num_samples_each_label', type=int, default=-1, metavar='NUM_SAMPLE',
                    help='num samples for each label, default: -1: all data.')

# model args
parser.add_argument('--model_backbone', type=str, default='resnet18', metavar='MODEL_BACKBONE',
                    choices=['resnet18', 'vit'],
                    help='Backbone of the feature extractor.')
parser.add_argument('--model_pretrained', action='store_true', help="Using pretrained model for learning or not.")
parser.add_argument('--cuda', type=int, default=0, metavar='CUDA',
                    help='The cuda device id, default to 0 '
                         'and specified with envir["CUDA_VISIBLE_DEVICES"].'
                         '-1 for using `cpu`.')
parser.add_argument('--vit_patch_size', type=int, default=16, metavar='VIT', help='')
parser.add_argument('--vit_dim', type=int, default=384, metavar='VIT', help='')
parser.add_argument('--vit_depth', type=int, default=9, metavar='VIT', help='')
parser.add_argument('--vit_heads', type=int, default=16, metavar='VIT', help='')
parser.add_argument('--vit_mlp_dim', type=int, default=1536, metavar='VIT', help='')
parser.add_argument('--vit_dropout', type=float, default=0.1, metavar='VIT', help='')
parser.add_argument('--vit_emb_dropout', type=float, default=0.1, metavar='VIT', help='')

# train args
parser.add_argument('--train_num_exp', type=int, default=-1, metavar='NUM',
                    help='Number of experiments to train in this run (default: -1 finish all exps).')
parser.add_argument('--train_mb_size', type=int, default=100, metavar='BS',
                    help='Number of images in a batch.')
parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01).')
parser.add_argument('--lr_schedule', type=str, default='cos', metavar='LR_SCHEDULE',
                    choices=['step', 'cos', 'none'],
                    help='learning rate schedule.')
parser.add_argument('--lr_schedule_step_size', type=int, default=20, metavar='STEP_SIZE',
                    help='step size in stepLR schedule.')
parser.add_argument('--lr_schedule_gamma', type=float, default=0.5, metavar='GAMMA',
                    help='gamma in stepLR schedule.')
parser.add_argument('--lr_schedule_eta_min', type=float, default=1e-6, metavar='ETA_MIN',
                    help='minimal lr in cosineAnn schedule.')
parser.add_argument('--n_experiences', type=int, default=10, metavar='NEXPERIENCES',
                    help='Number of tasks for continual training.')
parser.add_argument('--seed', type=int, default=1234, metavar='SEED',
                    help='Seed for training.')
parser.add_argument('--train_class_order', type=str, default='shuffle', metavar='SHUFFLE',
                    help='Option: [shuffle, fixed]. '
                         'If fixed, class_order will be chosen in FIXED_CLASS_ORDER for reproduction.')
parser.add_argument('--device', type=int, default=0, metavar='CUDA',
                    help='CUDA id use to train.')
parser.add_argument('--strategy', type=str, default='naive', metavar='STRATEGY',
                    help='Name of the used strategy.')

# # Augmentation parameters
# parser.add_argument('--color_jitter', type=float, default=0.3, metavar='PCT',
#                     help='Color jitter factor (default: 0.3)')
# parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
#                     help='Use AutoAugment policy. "v0" or "original". " + \
#                          "(default: rand-m9-mstd0.5-inc1)'),
# parser.add_argument('--train_interpolation', type=str, default='bicubic',
#                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
#
# # Random Erase params
# parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
#                     help='Random erase prob (default: 0.25)')
# parser.add_argument('--remode', type=str, default='pixel',
#                     help='Random erase mode (default: "pixel")')
# parser.add_argument('--recount', type=int, default=1,
#                     help='Random erase count (default: 1)')

# er args
parser.add_argument('--er_mem_size', type=int, default=1000, metavar='ER',
                    help='Memory buffer size to store past tasks.')

# lwf args
parser.add_argument('--lwf_alpha', type=float, default=1, metavar='LWF',
                    help='.')
parser.add_argument('--lwf_temperature', type=float, default=2, metavar='LWF',
                    help='.')

# gem args
parser.add_argument('--gem_patterns_per_exp', type=int, default=32, metavar='GEM',
                    help='.')
parser.add_argument('--gem_mem_strength', type=float, default=0.3, metavar='GEM',
                    help='.')

# ewc args
parser.add_argument('--ewc_lambda', type=float, default=1, metavar='EWC',
                    help='.')
parser.add_argument('--ewc_mode', type=str, default='separate', metavar='EWC',
                    help='.')
parser.add_argument('--ewc_decay', type=float, default=0.0, metavar='EWC',
                    help='Only used when ewc mode is `onlineweightedsum`.')

# our args
parser.add_argument('--ssc', type=float, default=0.1, metavar='REG',
                    help='SparseSelection coefficient.')
parser.add_argument('--scc', type=float, default=0.1, metavar='REG',
                    help='SupConLoss coefficient.')
parser.add_argument('--isc', type=float, default=0.0, metavar='REG',
                    help='IndependentSelection coefficient.')
parser.add_argument('--csc', type=float, default=0.0, metavar='REG',
                    help='ConsistentSelection coefficient.')

# evaluation during training
parser.add_argument('--eval_mb_size', type=int, default=50, metavar='BS',
                    help='Number of images in a batch.')
parser.add_argument('--eval_every', type=int, default=1, metavar='EVAL_EVERY',
                    help='Do evaluation every epochs for early stop.'
                         '-1 for not evaluation during epochs.')
parser.add_argument('--disable_early_stop', action='store_true', help="Do not use early stop.")
parser.add_argument('--eval_patience', type=int, default=5, metavar='PATIENCE',
                    help='Patience for EarlyStopingPlugin.')

# test args
parser.add_argument('--test_n_experiences', type=int, default=300, metavar='NEXPERIENCES',
                    help='Number of few-shot tasks.')
parser.add_argument('--test_n_way', type=int, default=10, metavar='WAY',
                    help='Number of way in few-shot tasks.')
parser.add_argument('--test_n_shot', type=int, default=10, metavar='SHOT',
                    help='Number of shot samples for each class in few-shot tasks.')
parser.add_argument('--test_n_val', type=int, default=5, metavar='VAL',
                    help='Number of val samples for each class in few-shot tasks.')
parser.add_argument('--test_n_query', type=int, default=10, metavar='QUERY',
                    help='Number of query samples for each class in few-shot tasks.')
parser.add_argument('--test_freeze_feature_extractor', action='store_true',
                    help='Freeze feature extractor when do few-shot testing.')

# path args
parser.add_argument('--pretrained_model_path', type=str, metavar='PATH',
                    default="../pretrained/pretrained_resnet.pt.tar",
                    help='Path of pretrained model.')
parser.add_argument('--dataset_root', type=str, metavar='PATH',
                    default="../datasets",
                    help='Path of dataset.')
parser.add_argument('--exp_root', type=str, metavar='PATH',
                    default="../avalanche-experiments",
                    help='Path of experiments.')

# log args
parser.add_argument('--use_wandb', action='store_true', help="Using wandb to visualize the results.")
parser.add_argument('--project_name', type=str, metavar='PROJECT_NAME',
                    default="CGQA",
                    help='Name of the project. CGQA for Compositional GQA project')
parser.add_argument('--exp_name', type=str, metavar='EXP_NAME',
                    default="TIME",
                    help='Name of the experiment. TIME for automatically assign according to the time.')
parser.add_argument('--use_interactive_logger', action='store_true',
                    help="Using interactive_logger.")
parser.add_argument('--do_not_store_checkpoint_per_exp', action='store_true',
                    help="Trigger do not store model{exp_id}.pth after each exp.")

# use for jupyter import args
parser.add_argument('-f', type=str, default="")

default_args = vars(parser.parse_args())

FIXED_CLASS_ORDER = {
    'continual':
        [40, 35, 81, 61, 98, 68, 85, 27, 39, 42,
         33, 59, 63, 94, 56, 87, 96,  1, 71, 82,
          9, 51, 29, 88, 75, 74, 62, 66, 79, 48,
          4, 64, 10, 93, 57, 72, 36,  7, 54, 77,
         21, 18, 70, 86, 22,  6, 44,  8, 41, 16,
         45, 20, 25, 55, 78, 31, 92,  5, 84, 32,
         52, 13, 91, 17, 28, 46, 60, 14, 65, 12,
         19,  2,  3,  0, 11, 67, 97, 34, 37, 95,
         50, 99, 73, 80, 69, 58, 90, 89, 43, 30,
         26, 23, 49, 15, 24, 76, 53, 38, 83, 47],
    # [[('fence', 'flower'), ('door', 'grass'), ('leaves', 'shirt'), ('grass', 'table'), ('shoe', 'shorts'),
    #   ('hat', 'table'), ('leaves', 'wall'), ('chair', 'grass'), ('door', 'shoe'), ('fence', 'helmet')],
    #  [('chair', 'sign'), ('grass', 'shorts'), ('hat', 'plate'), ('pole', 'shirt'), ('grass', 'pants'),
    #   ('pants', 'shoe'), ('pole', 'wall'), ('bench', 'chair'), ('helmet', 'plate'), ('leaves', 'shoe')],
    #  [('bench', 'shorts'), ('flower', 'pole'), ('chair', 'helmet'), ('pants', 'shorts'), ('helmet', 'shorts'),
    #   ('helmet', 'shoe'), ('hat', 'jacket'), ('hat', 'shorts'), ('jacket', 'shoe'), ('fence', 'wall')],
    #  [('bench', 'helmet'), ('hat', 'shirt'), ('bench', 'sign'), ('plate', 'wall'), ('grass', 'plate'),
    #   ('helmet', 'pole'), ('door', 'leaves'), ('bench', 'pants'), ('grass', 'jacket'), ('jacket', 'pole')],
    #  [('car', 'jacket'), ('building', 'plate'), ('helmet', 'leaves'), ('pants', 'shirt'), ('car', 'leaves'),
    #   ('bench', 'leaves'), ('fence', 'pants'), ('bench', 'shirt'), ('fence', 'grass'), ('building', 'jacket')],
    #  [('fence', 'plate'), ('car', 'helmet'), ('car', 'shorts'), ('grass', 'leaves'), ('jacket', 'shirt'),
    #   ('chair', 'shirt'), ('plate', 'sign'), ('bench', 'jacket'), ('leaves', 'sign'), ('chair', 'shoe')],
    #  [('flower', 'shirt'), ('building', 'chair'), ('plate', 'shorts'), ('building', 'leaves'), ('chair', 'hat'),
    #   ('fence', 'pole'), ('grass', 'sign'), ('building', 'grass'), ('hat', 'shoe'), ('bench', 'wall')],
    #  [('car', 'flower'), ('bench', 'door'), ('bench', 'hat'), ('bench', 'building'), ('bench', 'table'),
    #   ('hat', 'sign'), ('shirt', 'wall'), ('door', 'fence'), ('door', 'plate'), ('pole', 'table')],
    #  [('flower', 'pants'), ('shoe', 'sign'), ('helmet', 'shirt'), ('leaves', 'plate'), ('hat', 'wall'),
    #   ('grass', 'shoe'), ('plate', 'shirt'), ('pants', 'wall'), ('fence', 'leaves'), ('chair', 'pole')],
    #  [('car', 'sign'), ('car', 'pants'), ('flower', 'helmet'), ('building', 'hat'), ('car', 'shirt'),
    #   ('helmet', 'sign'), ('flower', 'wall'), ('door', 'pole'), ('leaves', 'shorts'), ('fence', 'shorts')]]
    'sys': None,
    'pro': None,
    'sub': None,
    'non': None,
    'noc': None,
    # for fresh and old concepts; 2-way; generated in datasets/process.ipynb
    'nonf': [26, 43, 67, 17, 49, 19, 76, 66, 70, 50, 81, 86, 22, 51, 84, 75, 66, 42, 25, 94, 26, 47, 75, 44, 15, 52, 76, 96, 67, 76, 36, 70, 83, 48, 15, 81, 17, 85, 76, 47, 85, 66, 96, 46, 36, 76, 73, 50, 26, 36, 51, 38, 46, 70, 48, 73, 84, 49, 88, 44, 86, 40, 43, 53, 84, 51, 15, 19, 51, 48, 89, 34, 66, 97, 83, 73, 19, 23, 76, 26, 53, 88, 19, 70, 34, 36, 64, 15, 66, 20, 64, 51, 89, 42, 81, 66, 15, 67, 96, 48, 51, 94, 89, 81, 25, 23, 94, 46, 97, 51, 64, 86, 53, 86, 88, 84, 69, 83, 44, 38, 17, 96, 96, 50, 46, 25, 34, 50, 49, 89, 69, 49, 85, 97, 23, 83, 20, 48, 75, 36, 81, 17, 66, 81, 53, 48, 26, 51, 43, 89, 50, 24, 73, 97, 49, 38, 94, 50, 22, 64, 40, 44, 36, 22, 19, 43, 38, 67, 86, 49, 48, 69, 36, 96, 46, 49, 15, 70, 40, 53, 72, 15, 69, 75, 49, 67, 25, 22, 66, 86, 52, 15, 38, 40, 94, 22, 36, 44, 97, 64, 15, 89, 66, 76, 17, 94, 36, 50, 52, 75, 88, 43, 72, 38, 38, 40, 83, 49, 67, 75, 85, 26, 46, 67, 20, 46, 48, 72, 83, 97, 34, 76, 20, 24, 23, 46, 53, 26, 96, 86, 20, 38, 83, 47, 20, 72, 86, 42, 84, 38, 69, 26, 52, 96, 24, 70, 46, 25, 19, 40, 66, 97, 81, 75, 94, 22, 38, 76, 38, 24, 47, 43, 89, 50, 94, 19, 88, 89, 38, 23, 38, 51, 81, 85, 40, 72, 64, 88, 48, 36, 43, 40, 96, 43, 64, 97, 19, 81, 89, 69, 76, 42, 96, 22, 23, 49, 47, 69, 47, 73, 49, 26, 66, 76, 23, 76, 42, 67, 53, 26, 94, 24, 23, 19, 46, 25, 70, 69, 94, 76, 88, 48, 48, 44, 85, 70, 50, 75, 47, 73, 94, 49, 66, 34, 49, 69, 75, 17, 44, 51, 75, 66, 23, 40, 46, 86, 97, 85, 96, 15, 42, 17, 84, 25, 23, 17, 75, 49, 66, 64, 24, 52, 53, 66, 89, 24, 36, 34, 43, 89, 22, 42, 94, 34, 85, 38, 64, 26, 46, 64, 17, 96, 64, 97, 46, 86, 64, 38, 66, 88, 25, 66, 50, 43, 34, 43, 26, 48, 51, 73, 94, 23, 96, 44, 83, 50, 97, 43, 43, 47, 25, 89, 69, 89, 36, 50, 70, 15, 20, 94, 69, 38, 53, 22, 76, 67, 24, 84, 50, 85, 83, 24, 48, 49, 81, 86, 24, 42, 22, 70, 85, 75, 86, 25, 75, 76, 38, 26, 83, 42, 15, 70, 67, 23, 84, 17, 17, 85, 76, 73, 25, 73, 43, 42, 49, 50, 49, 26, 44, 42, 43, 51, 83, 86, 25, 50, 34, 64, 69, 81, 25, 44, 40, 47, 76, 50, 52, 20, 20, 88, 72, 84, 42, 48, 49, 50, 73, 69, 42, 44, 48, 20, 34, 42, 34, 25, 36, 72, 36, 47, 52, 86, 73, 52, 42, 47, 84, 67, 89, 25, 83, 51, 53, 69, 97, 46, 19, 40, 23, 97, 81, 88, 69, 20, 48, 84, 69, 34, 73, 50, 76, 50, 19, 83, 73, 42, 89, 42, 15, 19, 76, 22, 36, 83, 97, 73, 75, 85, 44, 66, 40, 73, 83, 94, 66, 38, 24, 67, 22, 84, 69, 20, 22, 75, 70, 42, 19, 48, 83, 48, 85, 47, 94, 44, 84, 15, 24, 50, 73, 66],
    # nonf class: [('building', 'hat'), ('building', 'leaves'), ('car', 'flower'), ('car', 'helmet'), ('car', 'leaves'), ('car', 'pants'), ('car', 'shirt'), ('car', 'shorts'), ('car', 'sign'), ('door', 'fence'), ('door', 'leaves'), ('door', 'pole'), ('fence', 'flower'), ('fence', 'helmet'), ('fence', 'leaves'), ('fence', 'pants'), ('fence', 'pole'), ('fence', 'shorts'), ('fence', 'wall'), ('flower', 'helmet'), ('flower', 'pants'), ('flower', 'pole'), ('flower', 'shirt'), ('flower', 'wall'), ('hat', 'shirt'), ('hat', 'shorts'), ('hat', 'sign'), ('hat', 'wall'), ('helmet', 'leaves'), ('helmet', 'pole'), ('helmet', 'shirt'), ('helmet', 'shorts'), ('helmet', 'sign'), ('leaves', 'shirt'), ('leaves', 'shorts'), ('leaves', 'sign'), ('leaves', 'wall'), ('pants', 'shirt'), ('pants', 'shorts'), ('pants', 'wall'), ('pole', 'shirt'), ('pole', 'wall'), ('shirt', 'wall')]
    'nono': [58, 11, 58, 27, 58, 5, 32, 79, 61, 1, 61, 11, 58, 11, 54, 1, 1, 58, 27, 5, 61, 32, 79, 32, 54, 27, 1, 61, 5, 57, 27, 54, 32, 54, 57, 79, 61, 11, 58, 54, 11, 57, 61, 58, 11, 32, 1, 32, 79, 32, 79, 57, 11, 79, 61, 32, 57, 61, 5, 11, 11, 54, 1, 79, 1, 61, 57, 58, 58, 32, 58, 1, 79, 54, 5, 54, 27, 32, 27, 57, 32, 58, 5, 57, 1, 58, 57, 5, 32, 1, 61, 11, 1, 27, 1, 32, 57, 5, 54, 61, 11, 1, 1, 57, 32, 58, 27, 58, 58, 11, 1, 58, 32, 11, 27, 11, 58, 79, 54, 57, 54, 32, 57, 5, 1, 61, 1, 58, 32, 1, 58, 11, 79, 1, 1, 11, 11, 1, 57, 27, 27, 1, 57, 32, 79, 1, 11, 79, 32, 54, 61, 11, 27, 58, 1, 5, 5, 61, 79, 11, 27, 1, 57, 32, 57, 11, 79, 1, 57, 27, 54, 79, 5, 32, 57, 11, 1, 32, 61, 54, 57, 32, 57, 58, 32, 79, 5, 61, 61, 58, 57, 1, 57, 1, 1, 61, 11, 27, 57, 1, 5, 61, 61, 5, 27, 61, 1, 58, 27, 32, 58, 5, 27, 1, 58, 57, 27, 58, 5, 58, 32, 5, 79, 5, 5, 58, 1, 11, 27, 57, 1, 54, 61, 58, 61, 54, 57, 27, 61, 27, 32, 58, 79, 58, 1, 58, 61, 27, 32, 11, 5, 61, 27, 79, 32, 57, 27, 61, 1, 5, 58, 11, 79, 32, 54, 5, 57, 27, 11, 5, 5, 54, 32, 5, 27, 32, 5, 32, 54, 11, 32, 61, 79, 54, 61, 27, 58, 57, 5, 79, 32, 58, 61, 57, 58, 61, 58, 61, 61, 58, 58, 5, 54, 79, 57, 61, 57, 5, 54, 58, 61, 1, 58, 54, 32, 79, 79, 58, 54, 58, 32, 11, 54, 1, 58, 54, 32, 11, 5, 54, 5, 54, 54, 79, 54, 11, 11, 54, 32, 11, 32, 79, 32, 58, 11, 61, 1, 11, 27, 79, 54, 79, 5, 79, 79, 32, 1, 58, 57, 11, 32, 57, 58, 32, 5, 27, 5, 61, 58, 5, 1, 11, 54, 61, 57, 54, 1, 54, 32, 27, 57, 1, 32, 5, 11, 58, 79, 58, 5, 58, 27, 32, 5, 79, 79, 57, 5, 27, 11, 1, 11, 32, 79, 27, 32, 54, 79, 1, 32, 58, 79, 61, 1, 54, 1, 11, 54, 27, 11, 5, 1, 32, 27, 11, 11, 61, 79, 5, 57, 5, 58, 27, 5, 54, 79, 58, 32, 79, 27, 1, 61, 1, 27, 1, 54, 32, 57, 11, 58, 79, 32, 27, 79, 61, 1, 79, 27, 5, 32, 61, 11, 58, 11, 79, 32, 1, 79, 54, 54, 32, 58, 27, 57, 32, 58, 54, 11, 58, 58, 11, 11, 5, 57, 58, 58, 1, 32, 1, 61, 58, 79, 61, 58, 61, 5, 32, 5, 79, 54, 11, 61, 5, 27, 79, 57, 27, 11, 54, 58, 54, 61, 11, 11, 54, 1, 32, 11, 61, 5, 11, 11, 79, 61, 11, 79, 61, 57, 61, 1, 79, 61, 32, 1, 61, 1, 54, 1, 5, 1, 54, 11, 27, 27, 58, 57, 11, 58, 61, 58, 32, 61, 1, 79, 61, 57, 79, 79, 61, 5, 79, 54, 11, 11, 5, 5, 79, 11, 1, 58, 61, 32, 54, 11, 61, 5, 11, 79, 32, 61, 5, 79, 11, 79, 61, 58, 61, 1, 54, 58, 79, 11, 54, 57, 27, 11, 27, 11, 1, 27, 57],
    # nono class: [('bench', 'chair'), ('bench', 'jacket'), ('bench', 'table'), ('chair', 'grass'), ('chair', 'shoe'), ('grass', 'jacket'), ('grass', 'plate'), ('grass', 'shoe'), ('grass', 'table'), ('jacket', 'shoe')]
    'sysf': [19, 98, 7, 52, 47, 8, 65, 64, 91, 9, 78, 88, 10, 57, 78, 65, 54, 35, 98, 45, 6, 9, 88, 91, 98, 17, 24, 54, 10, 24, 37, 8, 77, 44, 6, 21, 7, 88, 65, 49, 36, 54, 47, 17, 91, 7, 37, 11, 36, 21, 35, 77, 87, 9, 7, 19, 52, 64, 47, 42, 16, 14, 35, 16, 36, 49, 35, 52, 80, 8, 8, 91, 52, 98, 49, 54, 41, 52, 90, 64, 8, 35, 65, 19, 6, 41, 11, 17, 19, 11, 16, 36, 39, 42, 96, 36, 78, 44, 7, 77, 41, 19, 8, 49, 96, 6, 68, 40, 91, 39, 64, 12, 11, 42, 45, 44, 36, 24, 47, 88, 65, 66, 42, 47, 65, 6, 9, 42, 78, 65, 80, 7, 12, 37, 36, 96, 9, 36, 65, 17, 9, 54, 68, 90, 45, 65, 41, 16, 98, 44, 6, 10, 88, 44, 11, 7, 42, 54, 44, 8, 87, 21, 12, 81, 56, 64, 7, 39, 9, 45, 19, 24, 39, 91, 54, 96, 10, 17, 39, 44, 6, 57, 68, 45, 65, 6, 57, 98, 42, 56, 14, 10, 54, 96, 47, 6, 21, 24, 56, 10, 44, 47, 90, 77, 6, 88, 65, 66, 7, 24, 42, 16, 65, 81, 41, 11, 6, 35, 81, 78, 65, 91, 45, 8, 47, 77, 78, 77, 96, 39, 88, 44, 80, 14, 80, 91, 17, 77, 9, 12, 11, 39, 64, 16, 45, 96, 9, 21, 78, 37, 19, 40, 98, 37, 36, 41, 12, 16, 14, 44, 8, 39, 35, 80, 52, 11, 57, 54, 90, 9, 11, 96, 52, 96, 57, 14, 12, 17, 41, 81, 88, 44, 80, 87, 35, 66, 54, 87, 54, 40, 78, 12, 19, 77, 49, 24, 64, 39, 44, 36, 77, 42, 9, 11, 91, 54, 87, 12, 56, 7, 39, 8, 49, 44, 49, 6, 88, 45, 98, 78, 77, 11, 11, 52, 24, 10, 42, 87, 98, 54, 45, 11, 81, 12, 98, 9, 19, 77, 12, 52, 35, 87, 66, 16, 47, 78, 6, 19, 90, 17, 65, 16, 11, 17, 47, 87, 21, 45, 24, 49, 68, 36, 8, 68, 12, 16, 96, 91, 96, 6, 35, 7, 80, 14, 81, 39, 14, 81, 10, 78, 12, 91, 98, 49, 44, 11, 17, 52, 36, 90, 10, 35, 91, 17, 87, 21, 80, 16, 81, 17, 39, 68, 8, 65, 78, 49, 47, 41, 96, 47, 96, 14, 49, 90, 17, 36, 9, 45, 10, 54, 40, 19, 6, 77, 81, 17, 9, 96, 36, 80, 19, 90, 36, 42, 10, 42, 78, 56, 56, 42, 80, 16, 81, 7, 42, 16, 11, 21, 57, 21, 8, 40, 87, 52, 91, 56, 39, 98, 44, 52, 49, 7, 66, 41, 21, 81, 78, 24, 80, 54, 7, 49, 54, 52, 16, 40, 24, 77, 7, 56, 78, 17, 12, 19, 14, 37, 65, 8, 17, 87, 68, 16, 39, 64, 10, 64, 64, 44, 44, 57, 40, 68, 57, 78, 41, 39, 24, 19, 57, 81, 47, 90, 45, 98, 96, 68, 88, 54, 42, 9, 37, 39, 8, 98, 65, 21, 88, 65, 88, 90, 9, 21, 66, 57, 45, 7, 90, 12, 90, 11, 49, 68, 41, 45, 64, 90, 10, 81, 11, 66, 42, 77, 78, 37, 47, 39, 40, 96, 81, 7, 52, 47, 96, 90, 80, 40, 68, 9, 8, 54, 88, 14, 91, 35, 19, 8, 45, 90, 39, 64, 96, 81, 68, 87, 65, 37, 77, 98, 49, 90, 77, 54, 45, 40, 68, 52, 7, 78, 19, 9],
    # sysf class: [('building', 'car'), ('building', 'door'), ('building', 'fence'), ('building', 'helmet'), ('building', 'pants'), ('building', 'shirt'), ('building', 'shorts'), ('building', 'wall'), ('car', 'door'), ('car', 'fence'), ('car', 'hat'), ('car', 'pole'), ('car', 'wall'), ('door', 'flower'), ('door', 'hat'), ('door', 'helmet'), ('door', 'pants'), ('door', 'shirt'), ('door', 'shorts'), ('door', 'sign'), ('door', 'wall'), ('fence', 'hat'), ('fence', 'shirt'), ('fence', 'sign'), ('flower', 'hat'), ('flower', 'leaves'), ('flower', 'shorts'), ('flower', 'sign'), ('hat', 'helmet'), ('hat', 'leaves'), ('hat', 'pants'), ('helmet', 'pants'), ('leaves', 'pants'), ('leaves', 'pole'), ('pants', 'pole'), ('pants', 'sign'), ('pole', 'shorts'), ('pole', 'sign'), ('shirt', 'shorts'), ('shirt', 'sign'), ('shorts', 'wall'), ('sign', 'wall')]
    'syso': [84, 5, 28, 93, 84, 3, 93, 33, 93, 5, 75, 5, 31, 2, 3, 84, 84, 31, 93, 31, 33, 28, 75, 2, 5, 28, 3, 75, 28, 93, 93, 85, 84, 75, 2, 84, 75, 84, 5, 31, 85, 2, 75, 2, 31, 72, 5, 85, 84, 31, 72, 5, 5, 33, 33, 2, 33, 84, 75, 31, 85, 33, 75, 3, 5, 28, 28, 72, 75, 85, 3, 75, 75, 5, 84, 72, 28, 85, 2, 85, 2, 31, 84, 3, 28, 85, 75, 3, 3, 5, 2, 28, 93, 84, 84, 5, 5, 93, 75, 85, 33, 84, 72, 75, 2, 33, 2, 85, 93, 5, 85, 3, 85, 5, 75, 31, 75, 28, 72, 31, 93, 2, 93, 5, 93, 85, 2, 93, 5, 75, 75, 3, 85, 5, 84, 93, 31, 85, 93, 5, 3, 2, 84, 28, 2, 28, 93, 72, 72, 93, 5, 33, 72, 31, 84, 72, 31, 5, 3, 2, 72, 85, 5, 2, 5, 84, 31, 33, 5, 33, 3, 31, 33, 3, 84, 33, 2, 28, 84, 5, 75, 3, 85, 2, 84, 72, 28, 85, 3, 93, 33, 31, 72, 93, 75, 72, 85, 93, 31, 2, 72, 28, 33, 84, 33, 31, 31, 75, 2, 72, 33, 5, 31, 5, 3, 84, 72, 28, 2, 31, 85, 93, 28, 33, 93, 31, 3, 2, 28, 85, 3, 28, 72, 3, 28, 31, 3, 85, 72, 5, 72, 85, 5, 33, 84, 28, 31, 93, 72, 31, 84, 72, 84, 85, 84, 93, 85, 72, 75, 3, 3, 85, 33, 3, 85, 3, 31, 84, 75, 33, 31, 85, 75, 85, 72, 84, 31, 5, 28, 33, 31, 5, 3, 72, 93, 3, 85, 5, 5, 33, 31, 5, 31, 33, 5, 93, 2, 5, 28, 85, 93, 31, 3, 33, 2, 72, 72, 5, 31, 72, 31, 85, 3, 28, 3, 85, 75, 3, 5, 2, 85, 5, 72, 75, 31, 28, 75, 2, 31, 3, 28, 75, 93, 2, 28, 93, 84, 33, 84, 33, 3, 84, 2, 72, 28, 2, 93, 33, 85, 28, 85, 84, 2, 28, 84, 2, 28, 2, 5, 93, 2, 31, 5, 28, 33, 3, 93, 3, 3, 72, 28, 31, 3, 28, 72, 33, 85, 72, 28, 85, 85, 2, 33, 31, 72, 5, 93, 75, 93, 28, 84, 3, 85, 84, 33, 93, 5, 75, 2, 93, 75, 31, 3, 93, 31, 72, 84, 75, 31, 5, 85, 84, 5, 33, 72, 84, 72, 33, 84, 93, 93, 2, 85, 75, 2, 85, 93, 31, 33, 3, 31, 85, 3, 28, 85, 93, 75, 93, 28, 5, 85, 5, 33, 5, 85, 5, 3, 5, 93, 28, 72, 5, 72, 28, 85, 84, 33, 31, 33, 93, 2, 28, 85, 72, 33, 72, 84, 85, 33, 93, 75, 5, 75, 31, 72, 85, 3, 5, 93, 2, 31, 33, 3, 5, 72, 5, 3, 85, 33, 5, 28, 75, 31, 33, 5, 85, 3, 5, 75, 33, 33, 84, 31, 84, 72, 85, 93, 33, 28, 93, 5, 33, 2, 93, 28, 5, 2, 85, 28, 84, 93, 75, 28, 93, 84, 85, 93, 85, 3, 72, 31, 2, 28, 93, 85, 2, 75, 3, 93, 5, 28, 2, 84, 28, 93, 5, 75, 93, 2, 3, 93, 85, 75, 28, 28, 84, 2, 93, 28, 3, 72, 2, 84, 31, 31, 75, 28, 33, 3, 75, 84, 75, 3, 72, 84, 85, 33, 28, 2, 5, 28, 72, 93, 84, 93, 5, 33, 28, 85, 33, 75, 5, 3, 72, 28, 5, 28, 2, 85, 33],
    # syso class: [('bench', 'grass'), ('bench', 'plate'), ('bench', 'shoe'), ('chair', 'jacket'), ('chair', 'plate'), ('chair', 'table'), ('jacket', 'plate'), ('jacket', 'table'), ('plate', 'shoe'), ('plate', 'table'), ('shoe', 'table')]
}
