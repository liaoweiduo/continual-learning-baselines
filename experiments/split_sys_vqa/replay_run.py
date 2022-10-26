"""
This script shows how to run an experiment on a specific strategy and benchmark.
You can override default parameters by providing a dictionary as input to the method.
You can find all the parameters used by the experiment in the source file of the experiment.

Place yourself into the project root folder.
"""
import argparse

# select the experiment
from experiments.split_sys_vqa import er_ssysvqa

# run the experiment with custom parameters (do not provide arguments to use default parameters)
# synaptic_intelligence_smnist({'learning_rate': 1e-3, 'si_lambda': 1})

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
parser.add_argument("--use_wandb", action='store_true', help='True to use wandb.')
parser.add_argument("--exp_name", type=str, default='Naive')
args = parser.parse_args()

res, res_novel = er_ssysvqa(vars(args))

'''
EXPERIMENTS: 
python experiments/split_sys_vqa/replay_run.py --use_wandb --exp_name ER --cuda 0
'''
