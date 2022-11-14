################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 17-10-2022                                                             #
# Author(s): Weiduo Liao                                                       #
# E-mail: liaowd@mail.sustech.edu.cn                                           #
################################################################################
import os
from pathlib import Path
from typing import Optional, Sequence, Union, Any, Dict, List
import json

import numpy as np
from torchvision import transforms

from avalanche.benchmarks.classic.classic_benchmarks_utils import check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.utils import PathsDataset, AvalancheDataset, AvalancheSubset
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDatasetType

"""
Default transforms borrowed from MetaShift.
Image shape: (3, 224, 224). 
Imagenet normalization.
"""
_image_size = (224, 224)
_default_cgqa_train_transform = transforms.Compose(
    [
        transforms.Resize(_image_size),      # allow reshape but not equal scaling
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

_default_cgqa_eval_transform = transforms.Compose(
    [
        transforms.Resize(_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def SplitSysGQA(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_cgqa_train_transform,
        eval_transform: Optional[Any] = _default_cgqa_eval_transform,
        dataset_root: Union[str, Path] = None,
        novel_combination: Optional[bool] = False,
        num_samples_each_label: Optional[int] = None,
):
    """
    Creates a CL benchmark using the pre-processed GQA dataset.

    Please first download GQA dataset (Image Files 20.3G) in
    https://cs.stanford.edu/people/dorarad/gqa/download.html.
    Then unzip and place the folder under dataset_root/vqa folder.
    Also, please download our preprocessed json
    under dataset_root/vqa/sys_gqa_json folder.

    Folder structure:
    dataset_root \
        - vqa \
            - allImages \
                - images \
                    - IMAGE_FILES
            - sys_gqa_json \
                - comb_train.json
                - comb_test.json
                - novel_comb_train.json
                - novel_comb_test.json

    The original labels of classes are the sorted combination of all existing
    objects defined in json. E.g., "apple,banana".

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences in the current benchmark.
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
    :param novel_combination: Whether to test on novel combination of objects.
        Defaults to False, which means to train the model.
        If it is set to True, n_experiences is overridden to 1,
        since we want to test the model performance on novel combination,
        and do not need the CL setting.
    :param num_samples_each_label: Sample specific number of images
        for each class for each exp (replace=True) with a build-in seed
        to support balance training.
        If is None, all samples are used.
        Note that total train samples for novel comb are around 100-200.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if dataset_root is None:
        dataset_root = default_dataset_location("gqa")

    '''Get _train_set, _test_set, _label_info defined in json'''
    _train_set, _test_set, _label_info = _get_sys_gqa_datasets(dataset_root,
                                                               shuffle=True, seed=1234,
                                                               novel_combination=novel_combination,
                                                               num_samples_each_label=num_samples_each_label)
    _label_set, _map_tuple_label_to_int, _map_int_label_to_tuple = _label_info

    if novel_combination:   # for novel testing
        classes_order = list(_map_int_label_to_tuple.keys())    # [20, 21, 22, 23, 24]
        selected_classes_in_exp = []        # e.g.[[21, 23], [20, 21],...]
        classes_in_exp = []                 # [[20,21], [20,21],...] if class-IL, [[0,1], [0,1],...] if task-IL
        class_mappings = []
        task_labels = []
        '''Select 2 classes for each exp as a 2-way task'''
        rng = np.random.RandomState(seed)
        for exp_idx in range(n_experiences):
            selected_class_idxs = rng.choice(classes_order, 2, replace=False)
            selected_classes_in_exp.append(selected_class_idxs)
            if return_task_id:
                classes_in_exp.append([0, 1])
                class_mapping = np.array([None for _ in range(max(selected_class_idxs))])
                class_mapping[selected_class_idxs] = [0, 1]
                class_mappings.append(class_mapping)
                task_labels.append(10)      # all exp are with task_label 10 in task-IL
            else:
                classes_in_exp.append([20, 21])
                class_mapping = np.array([None for _ in range(max(selected_class_idxs)+1)])
                class_mapping[selected_class_idxs] = [20, 21]
                class_mappings.append(class_mapping)
                task_labels.append(0)

        _train_subsets = [
            AvalancheSubset(_train_set,
                            indices=np.where(np.isin(_train_set.targets, selected_classes_in_exp[exp_idx]))[0],
                            class_mapping=class_mappings[exp_idx],
                            task_labels=task_labels[exp_idx])
            for exp_idx in range(n_experiences)]
        _test_subsets = [
            AvalancheSubset(_test_set,
                            indices=np.where(np.isin(_test_set.targets, selected_classes_in_exp[exp_idx]))[0],
                            class_mapping=class_mappings[exp_idx],
                            task_labels=task_labels[exp_idx])
            for exp_idx in range(n_experiences)]

        _benchmark_instance = dataset_benchmark(
            train_datasets=_train_subsets,
            test_datasets=_test_subsets,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_type=AvalancheDatasetType.CLASSIFICATION
        )

        _benchmark_instance.original_classes_in_exp = selected_classes_in_exp

        # _benchmark_instance = dataset_benchmark(
        #     train_datasets=[_train_set for _ in range(n_experiences)],
        #     test_datasets=[_test_set],
        #     complete_test_set_only=True,
        #     train_transform=train_transform,
        #     eval_transform=eval_transform,
        #     dataset_type=AvalancheDatasetType.CLASSIFICATION
        # )
        # _benchmark_instance.n_classes = len(_label_set)

    else:   # for training
        _benchmark_instance = nc_benchmark(
            train_dataset=_train_set,
            test_dataset=_test_set,
            n_experiences=n_experiences,
            task_labels=return_task_id,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            seed=seed,
            class_ids_from_zero_from_first_exp=not return_task_id,
            class_ids_from_zero_in_each_exp=return_task_id,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    _benchmark_instance.original_label_set = _label_set
    # {('tree', 'window'), ('roof', 'sky'), ('grass', 'hair'), ('car', 'shirt'), ('sign', 'wall'), ('building', 'wall')}
    _benchmark_instance.original_map_tuple_label_to_int = _map_tuple_label_to_int
    # {('tree', 'window'): 0, ('roof', 'sky'): 1, ...
    _benchmark_instance.original_map_int_label_to_tuple = _map_int_label_to_tuple
    # {0: ('tree', 'window'), 1: ('roof', 'sky'), ...

    return _benchmark_instance


def SplitSubGQA(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_cgqa_train_transform,
        eval_transform: Optional[Any] = _default_cgqa_eval_transform,
        dataset_root: Union[str, Path] = None,
        novel_combination: Optional[bool] = False,
        num_samples_each_label: Optional[int] = None,
):
    """
    Creates a CL benchmark using the pre-processed GQA dataset.

    Please first download GQA dataset (Image Files 20.3G) in
    https://cs.stanford.edu/people/dorarad/gqa/download.html.
    Then unzip and place the folder under dataset_root/vqa folder.
    Also, please download our preprocessed json
    under dataset_root/vqa/sub_gqa_json folder.

    Folder structure:
    dataset_root \
        - vqa \
            - allImages \
                - images \
                    - IMAGE_FILES
            - sub_gqa_json \
                - attriJson \
                    - attri_comb_train.json
                    - attri_comb_test.json
                    - novel_attri_comb_train.json
                    - novel_attri_comb_test.json

    The original labels of classes are the objects defined in json. E.g., "sky".

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences in the current benchmark.
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
    :param novel_combination: Whether to test on novel combination of objects.
        Defaults to False, which means to train the model.
        If it is set to True, n_experiences is overridden to 1,
        since we want to test the model performance on novel combination,
        and do not need the CL setting.
    :param num_samples_each_label: Sample specific number of images
        for each class for each exp (replace=True) with a build-in seed
        to support balance training.
        If is None, all samples are used.
        Note that total train samples for novel comb are around 100-200.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if dataset_root is None:
        dataset_root = default_dataset_location("gqa")

    '''Get _train_set, _test_set, _label_info defined in json'''
    _train_set, _test_set, _label_info = _get_sub_gqa_datasets(dataset_root,
                                                               shuffle=True, seed=1234,
                                                               novel_combination=novel_combination,
                                                               num_samples_each_label=num_samples_each_label)
    _label_set, _map_tuple_label_to_int, _map_int_label_to_tuple, _map_int_label_to_attr = _label_info

    if novel_combination:   # for novel testing
        # _benchmark_instance = dataset_benchmark(
        #     train_datasets=[_train_set for _ in range(n_experiences)],
        #     test_datasets=[_test_set],
        #     complete_test_set_only=True,
        #     train_transform=train_transform,
        #     eval_transform=eval_transform,
        #     dataset_type=AvalancheDatasetType.CLASSIFICATION
        # )
        # _benchmark_instance.n_classes = len(_label_set)

        _benchmark_instance = nc_benchmark(
            train_dataset=_train_set,
            test_dataset=_test_set,
            n_experiences=n_experiences,
            task_labels=return_task_id,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            seed=seed,
            class_ids_from_zero_from_first_exp=not return_task_id,
            class_ids_from_zero_in_each_exp=return_task_id,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    else:   # for training
        # if return_task_id:
        _benchmark_instance = nc_benchmark(
            train_dataset=_train_set,
            test_dataset=_test_set,
            n_experiences=n_experiences,
            task_labels=return_task_id,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            seed=seed,
            class_ids_from_zero_from_first_exp=not return_task_id,
            class_ids_from_zero_in_each_exp=return_task_id,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    _benchmark_instance.original_label_set = _label_set
    _benchmark_instance.original_map_tuple_label_to_int = _map_tuple_label_to_int
    _benchmark_instance.original_map_int_label_to_tuple = _map_int_label_to_tuple
    _benchmark_instance.original_map_int_label_to_attr = _map_int_label_to_attr

    return _benchmark_instance


def _get_sys_gqa_datasets(
        dataset_root,
        shuffle=True, seed: Optional[int] = None,
        novel_combination=False, num_samples_each_label=None,
        task_label=None):
    """
    Create systematicity GQA dataset, with given json files,
    containing instance tuples with shape (img_name, label, bounding box).

    If training (novel_combination=False), return the training comb. data,
    else, return only the novel comb. data.

    If novel_combination is True, you may need to specify task_label in Task-IL setting,
    since it is used after several train tasks.

    :param dataset_root: Path to the dataset root folder.
    :param shuffle: If true, the train sample order in the incremental experiences is
        randomly shuffled. Default to True.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param novel_combination: Whether to return novel comb.
        for evaluating model's compositional generalization performance.
    :param num_samples_each_label: If specify a certain number of samples for each label,
        random sampling with replace=True is used to sample for training.
        For evaluating, still all samples are used.
    :param task_label: if specify with int number, we will assign the task_label to data.
    :return data_sets defined by json file.
    """
    label_map_offset = 0
    if novel_combination:
        train_json_path = os.path.join(dataset_root, "gqa", "sys_gqa_json", "novel_comb_train.json")
        test_json_path = os.path.join(dataset_root, "gqa", "sys_gqa_json", "novel_comb_test.json")
        if task_label is None:      # class_incremental setting
            label_map_offset = 20       # novel comb's label starts from 20
    else:
        train_json_path = os.path.join(dataset_root, "gqa", "sys_gqa_json", "comb_train.json")
        test_json_path = os.path.join(dataset_root, "gqa", "sys_gqa_json", "comb_test.json")
    img_folder_path = os.path.join(dataset_root, "gqa", "allImages", "images")

    '''load image paths with labels and boundingBox'''
    with open(train_json_path, 'r') as f:
        train_img_info = json.load(f)
    with open(test_json_path, 'r') as f:
        test_img_info = json.load(f)
    # img_info:
    # [{'image': '2321003', 'label': ['shirt', 'wall'], 'boundingBox': [2, 4, 335, 368]},...

    '''preprocess labels to integers'''
    label_set = sorted(list(set([tuple(sorted(item['label'])) for item in test_img_info])))
    # [('building', 'sign'), ('building', 'sky'), ('building', 'window'), ('car', 'sign'), ('car', 'window'),
    #  ('grass', 'shirt'), ('grass', 'sky'), ('grass', 'tree'), ('hair', 'shirt'), ('hair', 'wall'), ('shirt', 'sign'),
    #  ('shirt', 'tree'), ('shirt', 'wall'), ('sign', 'sky'), ('sign', 'tree'), ('sign', 'wall'), ('sign', 'window'),
    #  ('sky', 'tree'), ('sky', 'window'), ('wall', 'window')]
    # or
    # [('building', 'hair'), ('car', 'sky'), ('grass', 'sign'), ('shirt', 'window'), ('tree', 'wall')]
    map_tuple_label_to_int = dict((item, idx + label_map_offset) for idx, item in enumerate(label_set))
    # {('building', 'sign'): 0, ('building', 'sky'): 1, ('building', 'window'): 2, ('car', 'sign'): 3,
    #  ('car', 'window'): 4, ('grass', 'shirt'): 5, ('grass', 'sky'): 6, ('grass', 'tree'): 7, ('hair', 'shirt'): 8,
    #  ('hair', 'wall'): 9, ('shirt', 'sign'): 10, ('shirt', 'tree'): 11, ('shirt', 'wall'): 12, ('sign', 'sky'): 13,
    #  ('sign', 'tree'): 14, ('sign', 'wall'): 15, ('sign', 'window'): 16, ('sky', 'tree'): 17, ('sky', 'window'): 18,
    #  ('wall', 'window'): 19}
    # or
    # {('building', 'hair'): 20, ('car', 'sky'): 21, ('grass', 'sign'): 22,
    #  ('shirt', 'window'): 23, ('tree', 'wall'): 24}
    map_int_label_to_tuple = dict((idx + label_map_offset, item) for idx, item in enumerate(label_set))
    # {0: ('building', 'sign'), 1: ('building', 'sky'),...
    # or
    # {20: ('building', 'hair'),...

    for item in train_img_info:
        item['image'] = f"{item['image']}.jpg"
        item['label'] = map_tuple_label_to_int[tuple(sorted(item['label']))]
    for item in test_img_info:
        item['image'] = f"{item['image']}.jpg"
        item['label'] = map_tuple_label_to_int[tuple(sorted(item['label']))]

    '''if num_samples_each_label provided, sample images (replace=True) to balance each class for train set'''
    selected_train_images = []
    if num_samples_each_label is not None and num_samples_each_label > 0:
        build_in_seed = 1234
        build_in_rng = np.random.RandomState(seed=build_in_seed)
        imgs_each_label = dict()
        for item in train_img_info:
            label = item['label']
            if label in imgs_each_label:
                imgs_each_label[label].append(item)
            else:
                imgs_each_label[label] = [item]
        for label, imgs in imgs_each_label.items():
            selected_idxs = build_in_rng.choice(np.arange(len(imgs)), num_samples_each_label, replace=True)
            for idx in selected_idxs:
                selected_train_images.append(imgs[idx])
    else:
        selected_train_images = train_img_info

    '''for testing, all images are used'''
    selected_test_images = test_img_info

    '''generate train_list and test_list: list with img tuple (path, label, bounding box)'''
    selected_train_images: List[Dict[str, Union[str, int, List[int]]]]
    selected_test_images:  List[Dict[str, Union[str, int, List[int]]]]

    train_list = []
    for item in selected_train_images:
        instance_tuple = (item['image'], item['label'], item['boundingBox'])
        train_list.append(instance_tuple)
    test_list = []
    for item in selected_test_images:
        instance_tuple = (item['image'], item['label'], item['boundingBox'])
        test_list.append(instance_tuple)
    # [('2325499C73236.jpg', 0, [2, 4, 335, 368]), ('2369086C73237.jpg', 0, [2, 4, 335, 368]),...

    '''shuffle the train set'''
    if shuffle:
        rng = np.random.RandomState(seed=seed)
        order = np.arange(len(train_list))
        rng.shuffle(order)
        train_list = [train_list[idx] for idx in order]

    '''generate train_set and test_set using PathsDataset'''
    '''generate AvalancheDataset with specified task_labels if provided'''
    '''TBD: use TensorDataset if pre-loading in memory'''
    _train_set = PathsDataset(
        root=img_folder_path,
        files=train_list,
        transform=transforms.Resize([224, 224])  # allow reshape but not equal scaling
    )
    _test_set = PathsDataset(
        root=img_folder_path,
        files=test_list,
        transform=transforms.Resize([224, 224])
    )
    _train_set = AvalancheDataset(_train_set, task_labels=task_label)
    _test_set = AvalancheDataset(_test_set, task_labels=task_label)

    return _train_set, _test_set, (label_set, map_tuple_label_to_int, map_int_label_to_tuple)


def _get_sys_gqa_datasets_pre_processed(
        dataset_root,
        novel_combination=False, num_samples_each_label=None,
        task_label=None):
    """
    Instruction:    (Not recommended)
        Please first download our pre-processed datasets in
        https://drive.google.com/file/d/1BzCVLggVKGpi0oXeAOUj1cOCZjRmVVos/view?usp=sharing.
        Then unzip and rename folder "dataset" to "sys_gqa" and place the folder under dataset_root.

    If training (novel_combination=False), return the training comb. data,
    else, return only the novel comb. data.

    Use train/test split to obtain train_images and test_images

    If novel_combination is True, you may need to specify task_label, since it is used after several train tasks.
    """
    label_map_offset = 0
    if novel_combination:
        json_path = os.path.join(dataset_root, "sys_gqa", "test.json")
        if task_label is None:      # class_incremental setting
            label_map_offset = 20       # novel comb's label starts from 20
    else:
        json_path = os.path.join(dataset_root, "sys_gqa", "train.json")
    img_folder_path = os.path.join(dataset_root, "sys_gqa", "cropped")

    '''load image paths with labels'''
    with open(json_path, 'r') as f:
        img_info = json.load(f)
    # img_info:
    # [{'image': '2414601C1', 'label': ['shirt', 'hair']}, {'image': '2385188C2', 'label': ['shirt', 'hair']},...

    '''preprocess labels to integers'''
    label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
    # {list:20} [('building', 'car'), ('building', 'roof'), ('building', 'shirt'),...
    # or
    # {list:5} [('building', 'wall'), ('car', 'shirt'),...
    map_tuple_label_to_int = dict((item, idx + label_map_offset) for idx, item in enumerate(label_set))
    # {('building', 'car'): 0, ('building', 'roof'): 1, ('building', 'shirt'): 2,...
    # or
    # {('building', 'wall'): 20, ('car', 'shirt'): 21,...
    map_int_label_to_tuple = dict((idx + label_map_offset, item) for idx, item in enumerate(label_set))
    # {0: ('building', 'car'), 1: ('building', 'roof'), 2: ('building', 'shirt'),...
    # or
    # {20: ('building', 'wall'), 21: ('car', 'shirt'),...

    '''sample images (replace=True) to balance each class, if shot provided'''
    for item in img_info:
        item['image'] = f"{item['image']}.jpg"
        item['label'] = map_tuple_label_to_int[tuple(sorted(item['label']))]

    selected_images = []
    if num_samples_each_label is not None and num_samples_each_label > 0:
        build_in_seed = 1234
        rng = np.random.RandomState(seed=build_in_seed)
        imgs_each_label = dict()
        for item in img_info:
            label = item['label']
            if label in imgs_each_label:
                imgs_each_label[label].append(item)
            else:
                imgs_each_label[label] = [item]
        for label, imgs in imgs_each_label.items():
            selected_idxs = rng.choice(np.arange(len(imgs)), num_samples_each_label, replace=True)
            for idx in selected_idxs:
                selected_images.append(imgs[idx])
    else:
        selected_images = img_info

    '''path list'''
    path_list = dict()

    selected_images: List[Dict[str, Union[str, int]]]

    for item in selected_images:
        path = item['image']
        if item['label'] in path_list.keys():
            path_list[item['label']].append(path)
        else:
            path_list[item['label']] = [path]
    # {0: ['2352689C73032.jpg', '2410259C73033.jpg', ...
    #  1: ['2326425C73287.jpg', '2320342C73288.jpg', ...
    #  ...

    '''generate train_list and test_list with 0.8/0.2 split of each label'''
    _train_list = []
    _test_list = []
    for label, images in path_list.items():
        split_idx = int(0.8 * len(images))

        instance_tuples = [(img, label) for img in images[:split_idx]]
        _train_list.extend(instance_tuples)

        instance_tuples = [(img, label) for img in images[split_idx:]]
        _test_list.extend(instance_tuples)
    # [('2325499C73236.jpg', 0), ('2369086C73237.jpg', 0),
    #  ('2405516C73238.jpg', 0), ('2316321C73239.jpg', 0),...

    '''generate train_set and test_set using PathsDataset'''
    '''generate AvalancheDataset with specified task_labels if provided'''
    '''TBD: use TensorDataset if pre-loading in memory'''
    _train_set = PathsDataset(
        root=img_folder_path,
        files=_train_list,
        transform=transforms.Resize([224, 224])  # allow reshape but not equal scaling
    )
    _test_set = PathsDataset(
        root=img_folder_path,
        files=_test_list,
        transform=transforms.Resize([224, 224])
    )
    _train_set = AvalancheDataset(_train_set, task_labels=task_label)
    _test_set = AvalancheDataset(_test_set, task_labels=task_label)

    return _train_set, _test_set, (label_set, map_tuple_label_to_int, map_int_label_to_tuple)


def _get_sub_gqa_datasets(
        dataset_root,
        shuffle=True, seed: Optional[int] = None,
        novel_combination=False, num_samples_each_label=None,
        task_label=None):
    """
    Create substitutivity GQA dataset, with given json files,
    containing instance tuples with shape (img_name, label, bounding box).

    If training (novel_combination=False), return the training comb. data,
    else, return only the novel comb. data.

    If novel_combination is True, you may need to specify task_label in Task-IL setting,
    since it is used after several train tasks.

    :param dataset_root: Path to the dataset root folder.
    :param shuffle: If true, the train sample order in the incremental experiences is
        randomly shuffled. Default to True.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param novel_combination: Whether to return novel comb.
        for evaluating model's compositional generalization performance.
    :param num_samples_each_label: If specify a certain number of samples for each label,
        random sampling with replace=True is used to sample for training.
        For evaluating, still all samples are used.
    :param task_label: if specify with int number, we will assign the task_label to data.
    :return data_sets defined by json file.
    """
    if novel_combination:
        train_json_path = os.path.join(dataset_root, "gqa", "sub_gqa_json", "attriJson", "novel_attri_comb_train.json")
        test_json_path = os.path.join(dataset_root, "gqa", "sub_gqa_json", "attriJson", "novel_attri_comb_test.json")
    else:
        train_json_path = os.path.join(dataset_root, "gqa", "sub_gqa_json", "attriJson", "attri_comb_train.json")
        test_json_path = os.path.join(dataset_root, "gqa", "sub_gqa_json", "attriJson", "attri_comb_test.json")
    img_folder_path = os.path.join(dataset_root, "gqa", "allImages", "images")

    '''load image paths with labels and boundingBox'''
    with open(train_json_path, 'r') as f:
        train_img_info = json.load(f)
    with open(test_json_path, 'r') as f:
        test_img_info = json.load(f)
    # img_info:
    # [{'image': '2321003', 'label': ['grass', 'green'], 'boundingBox': [2, 4, 335, 368]},...

    '''preprocess labels to integers'''
    label_set = sorted(list(set([item['label'][0] for item in test_img_info])))
    # ['building', 'car', 'chair', 'fence', 'flower', 'grass', 'hair', 'hat', 'helmet', 'jacket', 'pants', 'pole',
    #  'shirt', 'shoe', 'shorts', 'sign', 'sky', 'table', 'tree', 'wall']
    map_tuple_label_to_int = dict((item, idx) for idx, item in enumerate(label_set))
    # {'building': 0, 'car': 1, 'chair': 2, 'fence': 3, 'flower': 4, 'grass': 5, 'hair': 6, 'hat': 7, 'helmet': 8,
    #  'jacket': 9, 'pants': 10, 'pole': 11, 'shirt': 12, 'shoe': 13, 'shorts': 14, 'sign': 15, 'sky': 16, 'table': 17,
    #  'tree': 18, 'wall': 19}
    map_int_label_to_tuple = dict((idx, item) for idx, item in enumerate(label_set))
    # {0: 'building',...
    map_int_label_to_attr = dict()
    if novel_combination:       # we have test attribute specified for novel testing.
        tuple_label_attr = sorted(list(set([tuple(item['testComb']) for item in test_img_info])))
        # [('building', 'brown'), ('car', 'red'), ('chair', 'black'), ('fence', 'black'), ('flower', 'yellow'),
        #  ('grass', 'brown'), ('hair', 'black'), ('hat', 'blue'), ('helmet', 'white'), ('jacket', 'blue'),
        #  ('pants', 'white'), ('pole', 'wood'), ('shirt', 'green'), ('shoe', 'black'), ('shorts', 'blue'),
        #  ('sign', 'blue'), ('sky', 'white'), ('table', 'white'), ('tree', 'brown'), ('wall', 'brown')]
        map_int_label_to_attr = dict((map_tuple_label_to_int[label], attr) for label, attr in tuple_label_attr)
        # {0: 'brown', 1: 'red', 2: 'black', 3: 'black', 4: 'yellow', 5: 'brown', 6: 'black', 7: 'blue', 8: 'white',
        #  9: 'blue', 10: 'white', 11: 'wood', 12: 'green', 13: 'black', 14: 'blue', 15: 'blue', 16: 'white',
        #  17: 'white', 18: 'brown', 19: 'brown'}

    for item in train_img_info:
        item['image'] = f"{item['image']}.jpg"
        item['label'] = map_tuple_label_to_int[item['label'][0]]
    for item in test_img_info:
        item['image'] = f"{item['image']}.jpg"
        item['label'] = map_tuple_label_to_int[item['label'][0]]

    '''if num_samples_each_label provided, sample images (replace=True) to balance each class for train set'''
    selected_train_images = []
    if num_samples_each_label is not None and num_samples_each_label > 0:
        build_in_seed = 1234
        build_in_rng = np.random.RandomState(seed=build_in_seed)
        imgs_each_label = dict()
        for item in train_img_info:
            label = item['label']
            if label in imgs_each_label:
                imgs_each_label[label].append(item)
            else:
                imgs_each_label[label] = [item]
        for label, imgs in imgs_each_label.items():
            selected_idxs = build_in_rng.choice(np.arange(len(imgs)), num_samples_each_label, replace=True)
            for idx in selected_idxs:
                selected_train_images.append(imgs[idx])
    else:
        selected_train_images = train_img_info

    '''for testing, all images are used'''
    selected_test_images = test_img_info

    '''generate train_list and test_list: list with img tuple (path, label, bounding box)'''
    selected_train_images: List[Dict[str, Union[str, int, List[int]]]]
    selected_test_images:  List[Dict[str, Union[str, int, List[int]]]]

    train_list = []
    for item in selected_train_images:
        if novel_combination:
            instance_tuple = (item['image'], item['label'], item['boundingbox'])
        else:
            instance_tuple = (item['image'], item['label'], item['boundingBox'])
        train_list.append(instance_tuple)
    test_list = []
    for item in selected_test_images:
        if novel_combination:
            instance_tuple = (item['image'], item['label'], item['boundingbox'])
        else:
            instance_tuple = (item['image'], item['label'], item['boundingBox'])
        test_list.append(instance_tuple)
    # [('2325499C73236.jpg', 0, [2, 4, 335, 368]), ('2369086C73237.jpg', 0, [2, 4, 335, 368]),...

    '''shuffle the train set'''
    if shuffle:
        rng = np.random.RandomState(seed=seed)
        order = np.arange(len(train_list))
        rng.shuffle(order)
        train_list = [train_list[idx] for idx in order]

    '''generate train_set and test_set using PathsDataset'''
    '''generate AvalancheDataset with specified task_labels if provided'''
    '''TBD: use TensorDataset if pre-loading in memory'''
    _train_set = PathsDataset(
        root=img_folder_path,
        files=train_list,
        transform=transforms.Resize([224, 224])  # allow reshape but not equal scaling
    )
    _test_set = PathsDataset(
        root=img_folder_path,
        files=test_list,
        transform=transforms.Resize([224, 224])
    )
    _train_set = AvalancheDataset(_train_set, task_labels=task_label)
    _test_set = AvalancheDataset(_test_set, task_labels=task_label)

    label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple, map_int_label_to_attr)

    return _train_set, _test_set, label_info


if __name__ == "__main__":

    '''Sys'''
    # train_set, test_set, label_info = _get_sys_gqa_datasets(
    #     '../../datasets', novel_combination=False, num_samples_each_label=None, task_label=None)
    # train_set_novel, test_set_novel, label_info_novel = _get_sys_gqa_datasets(
    #     '../../datasets', shuffle=False, novel_combination=True)      #, num_samples_each_label=100, task_label=4)

    benchmark_instance = SplitSysGQA(n_experiences=10, return_task_id=False, seed=1234, shuffle=True,
                                     dataset_root='../../datasets')

    benchmark_novel = SplitSysGQA(n_experiences=10, return_task_id=False, seed=1234, shuffle=True,
                                  novel_combination=True,
                                  dataset_root='../../datasets')

    '''Sub'''
    # train_set, test_set, label_info = _get_sub_gqa_datasets(
    #     '../../datasets', novel_combination=False, num_samples_each_label=None, task_label=None)
    # train_set_novel, test_set_novel, label_info_novel = _get_sub_gqa_datasets(
    #     '../../datasets', shuffle=False, novel_combination=True)      #, num_samples_each_label=100, task_label=4)

    # benchmark_instance = SplitSubGQA(n_experiences=10, return_task_id=False, seed=1234, shuffle=True,
    #                                  dataset_root='../../datasets')

    # benchmark_novel = SplitSubGQA(n_experiences=10, return_task_id=False, seed=1234, shuffle=True,
    #                               novel_combination=True,
    #                               dataset_root='../../datasets')
    #
    # from torchvision.transforms import ToPILImage
    # from matplotlib import pyplot as plt
    # dataset = benchmark_instance.train_stream[0].dataset
    # x = dataset[0][0]
    # y = dataset[0][1]
    # # img = ToPILImage()(x)
    # img = x.numpy().transpose([1,2,0])
    # plt.figure()
    # plt.imshow(img)
    # plt.title(f'y:{y}')
    # plt.show()

    # check_vision_benchmark(benchmark_instance, show_without_transforms=True)
    # check_vision_benchmark(benchmark_novel, show_without_transforms=True)

__all__ = ["SplitSysGQA", "SplitSubGQA"]
