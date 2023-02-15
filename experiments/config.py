import argparse


parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--dataset', type=str, default='cgqa', choices=['cgqa', 'cpin'], metavar='DATASET',
                    help='Compositional GQA dataset: cgqa, Compositional PIN dataset: cpin.')
parser.add_argument('--dataset_mode', type=str, default='continual', metavar='DATASET_MODE',
                    choices=['continual', 'sys', 'pro', 'sub', 'non', 'noc'],
                    help='For cgqa, mode indicates the benchmark mode.')
parser.add_argument('--return_task_id', action='store_true',
                    help='Incremental setting: True for task-IL or False for class-IL.')

# model args
parser.add_argument('--model_backbone', type=str, default='resnet18', metavar='MODEL_BACKBONE',
                    choices=['resnet18', 'vit'],
                    help='Backbone of the feature extractor.')
parser.add_argument('--model_pretrained', action='store_true', help="Using pretrained model for learning or not.")
parser.add_argument('--cuda', type=int, default=0, metavar='CUDA',
                    help='The cuda device id, default to 0 '
                         'and specified with envir["CUDA_VISIBLE_DEVICES"].'
                         '-1 for using `cpu`.')

# train args
parser.add_argument('--train_mb_size', type=int, default=100, metavar='BS',
                    help='Number of images in a batch.')
parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01).')
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
# er args
parser.add_argument('--er_mem_size', type=int, default=1000, metavar='ER',
                    help='Memory buffer size to store past tasks.')

# lwf args
parser.add_argument('--lwf_alpha', type=float, default=1, metavar='LWF',
                    help='.')
parser.add_argument('--lwf_temperature', type=float, default=2, metavar='LWF',
                    help='.')

# gem args
parser.add_argument('--gem_patterns_per_exp', type=int, default=256, metavar='GEM',
                    help='.')
parser.add_argument('--gem_mem_strength', type=float, default=0.5, metavar='GEM',
                    help='.')

# ewc args
parser.add_argument('--ewc_lambda', type=float, default=1, metavar='EWC',
                    help='.')
parser.add_argument('--ewc_mode', type=str, default='separate', metavar='EWC',
                    help='.')
parser.add_argument('--ewc_decay', type=float, default=0.0, metavar='EWC',
                    help='Only used when ewc mode is `onlineweightedsum`.')

# evaluation during training
parser.add_argument('--eval_mb_size', type=int, default=50, metavar='BS',
                    help='Number of images in a batch.')
parser.add_argument('--eval_every', type=int, default=1, metavar='EVAL_EVERY',
                    help='Do evaluation every epochs for early stop.'
                         '-1 for not evaluation during epochs.')
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

parser.add_argument('-f', type=str, default="")    # use for jupyter import args

default_args = vars(parser.parse_args())

FIXED_CLASS_ORDER = {
    'continual':
        [[40, 35, 81, 61, 98, 68, 85, 27, 39, 42],
         [33, 59, 63, 94, 56, 87, 96,  1, 71, 82],
         [ 9, 51, 29, 88, 75, 74, 62, 66, 79, 48],
         [ 4, 64, 10, 93, 57, 72, 36,  7, 54, 77],
         [21, 18, 70, 86, 22,  6, 44,  8, 41, 16],
         [45, 20, 25, 55, 78, 31, 92,  5, 84, 32],
         [52, 13, 91, 17, 28, 46, 60, 14, 65, 12],
         [19,  2,  3,  0, 11, 67, 97, 34, 37, 95],
         [50, 99, 73, 80, 69, 58, 90, 89, 43, 30],
         [26, 23, 49, 15, 24, 76, 53, 38, 83, 47]],
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
}