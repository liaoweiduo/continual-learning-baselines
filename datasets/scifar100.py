
import os
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Dict, List
import json

import numpy as np
import torch
from torch.utils.data.dataset import Subset, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from avalanche.benchmarks.classic.classic_benchmarks_utils import check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.datasets.external_datasets.cifar import get_cifar100_dataset
from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.utils import PathsDataset, \
    classification_subset, concat_classification_datasets, make_classification_dataset


def _build_default_transform(image_size=(32, 32), is_train=True, normalize=True):
    """
    Default transforms borrowed from MetaShift.
    Imagenet normalization.
    """
    _train_transform = [
            transforms.Resize(image_size),  # allow reshape but not equal scaling
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    ]
    _eval_transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
    ]
    if normalize:
        _train_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))
        _eval_transform.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))

    _default_train_transform = transforms.Compose(_train_transform)
    _default_eval_transform = transforms.Compose(_eval_transform)

    if is_train:
        return _default_train_transform
    else:
        return _default_eval_transform


def build_transform_for_vit(img_size=(224, 224), is_train=True):
    if is_train:
        _train_transform = create_transform(
            input_size=img_size,
            is_training=is_train,
            color_jitter=0.3,  # 颜色抖动
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        # replace RandomResizedCropAndInterpolation with Resize, for not cropping img and missing concepts
        _train_transform.transforms[0] = transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC)

        return _train_transform
    else:
        return _build_default_transform(img_size, False)


def continual_training_benchmark(
        n_experiences: int,
        *,
        image_size=(128, 128),
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = None,
        eval_transform: Optional[Any] = None,
        dataset_root: Union[str, Path] = None,
        memory_size: int = 0,
        num_samples_each_label: Optional[int] = None,
        multi_task: bool = False,
):
    """
    Creates a CL benchmark using the pre-processed GQA dataset.

    :param n_experiences: The number of experiences in the current benchmark.
    :param image_size: size of image.
    :param return_task_id: If True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'tinyimagenet' will be used.
    :param memory_size: Total memory size for store all past classes/tasks.
        Each class has equal number of instances in the memory.
    :param num_samples_each_label: Number of samples for each label,
        -1 or None means all data are used.
    :param multi_task: if True, return a multi_task benchmark,
        else, return a continual learning benchmark.

    :returns: A properly initialized instance: `GenericCLScenario`
        with train_stream, val_stream, test_stream.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location("gqa")

    if train_transform is None:
        train_transform = _build_default_transform(image_size, True)
    if eval_transform is None:
        eval_transform = _build_default_transform(image_size, False)

    '''load datasets'''
    # if num_samples_each_label is None or num_samples_each_label < 0:
    #     num_samples_each_label = None
    num_samples_each_label = None

    train_set, test_set = get_cifar100_dataset(dataset_root)

    '''preprocess labels to integers'''
    label_set = sorted(list(set([item[1] for item in test_set])))
    label_offset = 0
    # [('building', 'sign'), ...]
    map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
    # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
    map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
    # {0: ('building', 'sign'), 1: ('building', 'sky'),...}
    label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    '''generate class order for continual tasks'''
    num_classes = len(label_set)
    assert num_classes % n_experiences == 0
    num_class_in_exp = num_classes // n_experiences
    classes_order = np.array(list(map_int_label_to_tuple.keys())).astype(np.int64)  # [0-99]
    if fixed_class_order is not None:
        assert len(fixed_class_order) == num_classes
        classes_order = np.array(fixed_class_order).astype(np.int64)
    elif shuffle:
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(classes_order)

    original_classes_in_exp = classes_order.reshape([n_experiences, num_class_in_exp])  # e.g.[[5, 2], [6, 10],...]
    if return_task_id:      # task-IL
        classes_in_exp = np.stack([np.arange(num_class_in_exp) for _ in range(n_experiences)]).astype(np.int64)
        # [[0,1], [0,1],...]
    else:
        classes_in_exp = np.arange(num_classes).reshape([n_experiences, num_class_in_exp]).astype(np.int64)
        # [[0,1], [2,3],...]

    '''class mapping for each exp, contain the mapping for previous exps (unseen filled with -1)'''
    '''so that it allow memory buffer for previous exps'''
    class_mappings = []
    for exp_idx in range(n_experiences):
        class_mapping = np.array([-1] * num_classes)
        class_mapping[original_classes_in_exp[:exp_idx+1].reshape(-1)] = classes_in_exp[:exp_idx+1].reshape(-1)
        class_mappings.append(class_mapping)    # [-1 -1  2 ... -1  6 -1 ... -1  0 -1 ... -1]
    class_mappings = np.array(class_mappings).astype(np.int64)

    '''get sample indices for each experiment'''
    rng = np.random.RandomState(seed)   # reset rng for memory selection

    def obtain_subset(dataset, exp_idx, memory_size=0):
        t = dataset.targets
        exp_classes = original_classes_in_exp[exp_idx]
        indices = [np.where(np.isin(t, exp_classes))[0]]    # current exp
        task_id = exp_idx if return_task_id else 0
        task_labels = [[task_id for _ in range(len(indices[0]))]]

        if memory_size > 0 and exp_idx > 0:
            old_classes = original_classes_in_exp[:exp_idx].reshape(-1)
            class_task_ids = {
                cls: t_id if return_task_id else 0
                for t_id, clses in enumerate(original_classes_in_exp[:exp_idx]) for cls in clses}

            num_instances_each_class = int(memory_size / len(old_classes))
            for cls in old_classes:
                cls_indices = np.where(t == cls)[0]
                rng.shuffle(cls_indices)
                indices.append(cls_indices[:num_instances_each_class])
                task_labels.append([class_task_ids[cls] for _ in range(len(indices[-1]))])

        indices = np.concatenate(indices)
        task_labels = np.concatenate(task_labels)
        assert indices.shape[0] == task_labels.shape[0]

        mapped_targets = np.array([class_mappings[exp_idx][idx] for idx in np.array(t)[indices]])
        return make_classification_dataset(
            MySubset(dataset, indices=list(indices), class_mapping=class_mappings[exp_idx]),
            targets=mapped_targets,
            task_labels=task_labels,
        )

    train_subsets = [
        obtain_subset(train_set, expidx, memory_size)
        for expidx in range(n_experiences)
    ]
    test_subsets = [
        obtain_subset(test_set, expidx)
        for expidx in range(n_experiences)
    ]

    if multi_task:
        train_subsets = [
            concat_classification_datasets(
                train_subsets,
        )]

    benchmark_instance = dataset_benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        other_streams_datasets={'val': train_subsets},
        train_transform=train_transform,
        eval_transform=eval_transform,
        other_streams_transforms={'val': (eval_transform, None)},
    )
    benchmark_instance.original_classes_in_exp = original_classes_in_exp
    benchmark_instance.classes_in_exp = classes_in_exp
    benchmark_instance.class_mappings = class_mappings
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info
    benchmark_instance.return_task_id = return_task_id

    return benchmark_instance


class MySubset(Subset):
    """
    subset with class mapping
    """
    def __init__(self, dataset, indices: list, class_mapping, transform=None):
        super().__init__(dataset, indices)
        # self._dataset = dataset
        # self._indices = indices
        # self._subset = Subset(dataset, indices)
        self._class_mapping = class_mapping
        self._transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self._transform is not None:
            x = self._transform(x)
        mapped_y = self._class_mapping[y]
        return x, mapped_y


__all__ = ["continual_training_benchmark"]


if __name__ == "__main__":
    benchmark = continual_training_benchmark(
        n_experiences=10, return_task_id=False,
        seed=1234, shuffle=True,
        dataset_root='../../datasets',
        memory_size=0,
        multi_task=True,
    )
