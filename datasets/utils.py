################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 28-10-2022                                                             #
# Author(s): Weiduo Liao                                                       #
# E-mail: liaowd@mail.sustech.edu.cn                                           #
################################################################################
""" This module handles all the functionalities related to the logging of
Avalanche experiments using Weights & Biases. """

from typing import Union, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torchvision import transforms

from avalanche.logging import WandBLogger


class WandBDatasetAnalyzer(WandBLogger):
    """Weights and Biases Dataset Analyzer

    """
    def __init__(
        self,
        project_name: str = "Avalanche",
        run_name: str = "dataset visualization",
        dir: Union[str, Path] = None,
        params: dict = None,
    ):
        super().__init__(project_name, run_name, dir=dir, params=params)

    def log_dataset(
            self,
            dataset,
            label_map: Dict[int, str],
            visual_name: str,
            num_samples_each_label: Optional[int] = None,
            image_size: Tuple[int, int] = (84, 84),
            seed=1234
    ):
        """
        Use log_dataset to analyze given dataset and log example image for visualization.
        """

        trans = transforms.Resize(image_size)

        '''Collect sample idxs'''
        samples = dict()
        print('Collect sample idxs.')
        for idx, label in enumerate(tqdm(dataset.targets)):
            if label in samples:
                samples[label].append(idx)
            else:
                samples[label] = [idx]

        '''Sort label'''
        samples = dict(sorted(samples.items()))

        '''Number of samples'''
        print('Calculate number of samples.')
        num_samples = {key: len(items) for key, items in samples.items()}

        '''Sample some images with transformation'''
        rng = np.random.RandomState(seed)
        if num_samples_each_label is not None:
            print('Select and load some samples.')
            selected_samples = {
                key: rng.choice(idxs, num_samples_each_label, replace=False) for key, idxs in samples.items()}
        else:
            selected_samples = samples

        for key, idxs in tqdm(selected_samples.items()):
            selected_samples[key] = [self.wandb.Image(trans(dataset[idx][0])) for idx in idxs]

        '''Construct wandb table'''
        data = [
            [label, label_map[label], images, num_samples[label]] for label, images in
            selected_samples.items()
        ]
        columns = ["Label", "Label string", "Images", "Num"]
        train_table = self.wandb.Table(data=data, columns=columns)

        '''Log wandb table'''
        self.wandb.log({visual_name: train_table})


if __name__ == '__main__':
    from cgqa import _get_sys_gqa_datasets

    _train_set, _test_set, label_info = _get_sys_gqa_datasets('../../datasets', shuffle=False, novel_combination=False)

    label_map = {key: ', '.join(item) for key, item in label_info[2].items()}
    # {0: 'building, sign', 1: 'building, sky',...

    wandb_dataset_analyzer = WandBDatasetAnalyzer(
        project_name="Split_Sys_VQA",
        run_name="Systematicity_GQA_datasets", dir='../../avalanche-experiments')

    wandb_dataset_analyzer.log_dataset(_train_set, label_map, visual_name="Train samples")  #, num_samples_each_label=200)
    wandb_dataset_analyzer.log_dataset(_test_set, label_map, visual_name="Test samples")

    _train_set, _test_set, label_info = _get_sys_gqa_datasets('../../datasets', shuffle=False, novel_combination=True)

    label_map = {key: ', '.join(item) for key, item in label_info[2].items()}
    # {20: 'building, hair', 1: 'car, sky',...

    # from matplotlib import pyplot as plt
    # x = test_set[1][0]
    # y = test_set[1][1]
    # img = x
    # plt.figure()
    # plt.imshow(img)
    # plt.title(f'y:{y}')
    # plt.show()

    wandb_dataset_analyzer.log_dataset(_train_set, label_map, visual_name="Novel Train samples")
    wandb_dataset_analyzer.log_dataset(_test_set, label_map, visual_name="Novel Test samples")
