################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-1-2023                                                              #
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

""" README
The original labels of classes are the sorted combination of all existing
objects defined in json. E.g., "apple,banana".
"""


"""
Default transforms borrowed from MetaShift.
Imagenet normalization.
"""
_image_size = (128, 128)
_default_cgqa_train_transform = transforms.Compose(
    [
        transforms.Resize(_image_size),  # allow reshape but not equal scaling
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


def continual_training_benchmark(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_cgqa_train_transform,
        eval_transform: Optional[Any] = _default_cgqa_eval_transform,
        dataset_root: Union[str, Path] = None,
        memory_size: int = 0,
):
    """
    Creates a CL benchmark using the pre-processed PIN dataset.

    List of 20 objects:
        [ 'Bovidae Foot', 'Bovidae Head', 'Bovidae Body', 'Canidae Head',
        'Canidae Foot', 'Canidae Body', 'Lacertilia Head', 'Lacertilia Body',
        'Lacertilia Foot', 'Primates Head', 'Primates Hand', 'Primates Body',
        'Primates Foot', 'Felidae Head', 'Felidae Body', 'Felidae Foot',
        'Mustelidae Head', 'Mustelidae Body', 'Mustelidae Foot',
        'Testudines Foot', 'Testudines Body', 'Testudines Head',
        'Ursidae Head', 'Ursidae Body', 'Ursidae Foot', 'Car Side Mirror',
        'Car Tier', 'Car Body', 'Fish Head',  'Fish Fin', 'Fish Body',
        'Snake Head', 'Snake Body', 'Bird Head', 'Bird Body', 'Bird Wing',
        'Bird Foot']

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
    :param memory_size: Total memory size for store all past classes/tasks.
        Each class has equal number of instances in the memory.

    :returns: A properly initialized instance: `GenericCLScenario`
        with train_stream, val_stream, test_stream.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location("pin")

    '''load datasets'''
    datasets, label_info = _get_pin_datasets(dataset_root, mode='continual')
    train_set, val_set, test_set = datasets['train'], datasets['val'], datasets['test']
    label_set, map_tuple_label_to_int, map_int_label_to_tuple = label_info

    '''generate class order for continual tasks'''
    num_classes = len(label_set)
    assert num_classes % n_experiences == 0
    num_class_in_exp = num_classes // n_experiences
    classes_order = np.array(list(map_int_label_to_tuple.keys()))  # [0-99]
    if fixed_class_order is not None:
        assert len(fixed_class_order) == num_classes
        classes_order = np.array(fixed_class_order)
    elif shuffle:
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(classes_order)

    original_classes_in_exp = classes_order.reshape([n_experiences, num_class_in_exp])  # e.g.[[5, 2], [6, 10],...]
    if return_task_id:      # task-IL
        classes_in_exp = np.stack([np.arange(num_class_in_exp) for _ in range(n_experiences)])  # [[0,1], [0,1],...]
    else:
        classes_in_exp = np.arange(num_classes).reshape([n_experiences, num_class_in_exp])  # [[0,1], [2,3],...]

    '''class mapping for each exp, contain the mapping for previous exps (unseen filled with -1)'''
    '''so that it allow memory buffer for previous exps'''
    class_mappings = []
    for exp_idx in range(n_experiences):
        class_mapping = np.array([-1] * num_classes)
        class_mapping[original_classes_in_exp[:exp_idx+1].reshape(-1)] = classes_in_exp[:exp_idx+1].reshape(-1)
        class_mappings.append(class_mapping)    # [-1 -1  2 ... -1  6 -1 ... -1  0 -1 ... -1]
    class_mappings = np.array(class_mappings)

    '''get sample indices for each experiment'''
    rng = np.random.RandomState(seed)   # reset rng for memory selection

    def obtain_subset(dataset, exp_idx, memory_size):
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
        return AvalancheSubset(
            dataset,
            indices=indices,
            class_mapping=class_mappings[exp_idx],
            transform_groups={'val': (None, None)},
            task_labels=task_labels)

    train_subsets = [
        obtain_subset(train_set, expidx, memory_size)
        for expidx in range(n_experiences)
    ]

    val_subsets = [
        obtain_subset(val_set, expidx, 0)
        for expidx in range(n_experiences)
    ]

    test_subsets = [
        obtain_subset(test_set, expidx, 0)
        for expidx in range(n_experiences)
    ]

    benchmark_instance = dataset_benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        other_streams_datasets={'val': val_subsets},
        train_transform=train_transform,
        eval_transform=eval_transform,
        other_streams_transforms={'val': (eval_transform, None)},
    )

    benchmark_instance.original_classes_in_exp = original_classes_in_exp
    benchmark_instance.classes_in_exp = classes_in_exp
    benchmark_instance.class_mappings = class_mappings
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info

    return benchmark_instance


def fewshot_testing_benchmark(
        n_experiences: int,
        *,
        n_way: int = 10,
        n_shot: int = 10,
        n_val: int = 5,
        n_query: int = 10,
        mode: str = 'sys',
        task_offset: int = 10,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = _default_cgqa_train_transform,
        eval_transform: Optional[Any] = _default_cgqa_eval_transform,
        dataset_root: Union[str, Path] = None,
):
    """
    Creates a CL benchmark using the pre-processed PIN dataset.

    For fewshot testing, you need to specify the specific testing mode.

    :param n_experiences: The number of experiences in the current benchmark.
        In the fewshot setting, it means the number of few-shot tasks
    :param n_way: Number of ways for few-shot tasks.
    :param n_shot: Number of support image instances for each class.
    :param n_val: Number of evaluation image instances for each class.
    :param n_query: Number of query image instances for each class.
    :param mode: Option [sys, pro, sub, non, noc].
    :param task_offset: Offset for tasks not start from 0 in task-IL.
        Default to 10 since continual training consists of 10 tasks.
        You need to specify to 1 for class-IL.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
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

    :returns: A properly initialized instance: `GenericCLScenario`.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location("pin")

    '''load datasets'''
    datasets, label_info = _get_pin_datasets(dataset_root, mode=mode)
    dataset = datasets['dataset']
    label_set, map_tuple_label_to_int, map_int_label_to_tuple = label_info

    '''generate fewshot tasks'''
    num_classes = len(label_set)
    classes_order = list(map_int_label_to_tuple.keys())  # [0-99]
    selected_classes_in_exp = []  # e.g.[[5, 4], [6, 5],...]
    classes_in_exp = []  # [[0,1], [0,1],...]
    class_mappings = []
    task_labels = []
    t = np.array(dataset.targets)
    train_subsets, val_subsets, test_subsets = [], [], []

    if fixed_class_order is not None:
        assert len(fixed_class_order) == n_experiences * n_way
        selected_classes_in_exp = np.array(fixed_class_order).reshape(n_experiences, n_way)
    else:
        rng = np.random.RandomState(seed=seed)
        for exp_idx in range(n_experiences):
            '''select n_way classes for each exp'''
            selected_class_idxs = rng.choice(classes_order, n_way, replace=False)
            selected_classes_in_exp.append(selected_class_idxs)

    for exp_idx in range(n_experiences):
        selected_class_idxs = selected_classes_in_exp[exp_idx]
        classes_in_exp.append(np.arange(n_way))
        class_mapping = np.array([-1] * num_classes)
        class_mapping[selected_class_idxs] = np.arange(n_way)
        class_mappings.append(class_mapping)
        task_labels.append(exp_idx + task_offset)

    rng = np.random.RandomState(seed)
    for exp_idx in range(n_experiences):
        selected_class_idxs = selected_classes_in_exp[exp_idx]
        '''select n_shot+n_val+n_query images for each class'''
        shot_indices, val_indices, query_indices = [], [], []
        for cls_idx in selected_class_idxs:
            indices = rng.choice(np.where(t == cls_idx)[0], n_shot + n_val + n_query, replace=False)
            shot_indices.append(indices[:n_shot])
            val_indices.append(indices[n_shot:n_shot+n_val])
            query_indices.append(indices[n_shot+n_val:])
        shot_indices = np.concatenate(shot_indices)
        val_indices = np.concatenate(val_indices)
        query_indices = np.concatenate(query_indices)
        train_subsets.append(
            AvalancheSubset(
                dataset,
                indices=shot_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )
        val_subsets.append(
            AvalancheSubset(
                dataset,
                indices=val_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )
        test_subsets.append(
            AvalancheSubset(
                dataset,
                indices=query_indices,
                class_mapping=class_mappings[exp_idx],
                transform_groups={'val': (None, None)},
                task_labels=task_labels[exp_idx])
        )

    benchmark_instance = dataset_benchmark(
        train_datasets=train_subsets,
        test_datasets=test_subsets,
        other_streams_datasets={'val': val_subsets},
        train_transform=train_transform,
        eval_transform=eval_transform,
        other_streams_transforms={'val': (eval_transform, None)},
    )

    benchmark_instance.original_classes_in_exp = np.array(selected_classes_in_exp)
    benchmark_instance.classes_in_exp = np.array(classes_in_exp)
    benchmark_instance.class_mappings = np.array(class_mappings)
    benchmark_instance.n_classes = num_classes
    benchmark_instance.label_info = label_info

    return benchmark_instance


def _get_pin_datasets(
        dataset_root,
        shuffle=False, seed: Optional[int] = None,
        mode='continual',
        num_samples_each_label=None,
        label_offset=0,
        preprocessed=True,
):
    """
    Create PIN dataset, with given json files,
    containing instance tuples with shape (img_name, label).

    You may need to specify label_offset if relative label do not start from 0.

    :param dataset_root: Path to the dataset root folder.
    :param shuffle: If true, the train sample order (in json)
        in the incremental experiences is
        randomly shuffled. Default to False.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param mode: Option [continual, sys, pro, non, noc].
    :param num_samples_each_label: If specify a certain number of samples for each label,
        random sampling (build-in seed:1234,
        and replace=True if num_samples_each_label > num_samples, else False)
        is used to sample.
        Only for continual mode, only apply to train dataset.
    :param label_offset: specified if relative label not start from 0.
    :param preprocessed (DISABLED): True, just load preprocessed images,
        specified by newImageName,
        while False, construct new image by the defined object list.
        Default True.

    :return data_sets defined by json file and label information.
    """
    img_folder_path = os.path.join(dataset_root, "PIN", "PIN")

    def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int):
        for item in img_info:
            item['image'] = item['newFileName']
            item['comb'] = item['label']
            item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['label']))]

    def split_img_info(img_info):
        split_img_info = {'train': [], 'val': [], 'test': []}
        for item in img_info:
            image_path = item['image']
            # "continual/train/1.jpg"
            split = image_path.split('/')[1]    # train/val/test
            assert (split in ['train', 'val', 'test']
                    ), f'wrong split: {split} for image: {image_path}.'
            split_img_info[split].append(item)

        return split_img_info['train'], split_img_info['val'], split_img_info['test']

    def formulate_img_tuples(images):
        """generate train_list and test_list: list with img tuple (path, label)"""
        img_tuples = []
        for item in images:
            instance_tuple = (item['image'], item['label'])     # , item['boundingBox']
            img_tuples.append(instance_tuple)
        return img_tuples

    if mode == 'continual':
        json_path = os.path.join(img_folder_path, "continual.json")

        with open(json_path, 'r') as f:
            img_info = json.load(f)
        # img_info:
        # [{"label": ["Lacertilia Body", "Fish Body", "Primates Head", "Lacertilia Head", "Bovidae Body"],
        #   "newFileName": "fewshot/non/119001.jpg",
        #   "objects": [...]
        #   "position": [1, 4, 0, 2, 5],...]

        '''preprocess labels to integers'''
        label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
        # [('building', 'sign'), ...]
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        # {0: ('building', 'sign'), 1: ('building', 'sky'),...}

        preprocess_label_to_integer(img_info, map_tuple_label_to_int)
        train_img_info, val_img_info, test_img_info = split_img_info(img_info)

        '''if num_samples_each_label provided, sample images to balance each class for train set'''
        selected_train_images = []
        if num_samples_each_label is not None and num_samples_each_label > 0:
            imgs_each_label = dict()
            for item in train_img_info:
                label = item['label']
                if label in imgs_each_label:
                    imgs_each_label[label].append(item)
                else:
                    imgs_each_label[label] = [item]
            build_in_seed = 1234
            build_in_rng = np.random.RandomState(seed=build_in_seed)
            for label, imgs in imgs_each_label.items():
                selected_idxs = build_in_rng.choice(
                    np.arange(len(imgs)), num_samples_each_label,
                    replace=True if num_samples_each_label > len(imgs) else False)
                for idx in selected_idxs:
                    selected_train_images.append(imgs[idx])
        else:
            selected_train_images = train_img_info

        '''generate train_list and test_list: list with img tuple (path, label)'''
        train_list = formulate_img_tuples(selected_train_images)
        val_list = formulate_img_tuples(val_img_info)
        test_list = formulate_img_tuples(test_img_info)
        # [('continual/val/59767.jpg', 0),...

        '''shuffle the train set'''
        if shuffle:
            rng = np.random.RandomState(seed=seed)
            order = np.arange(len(train_list))
            rng.shuffle(order)
            train_list = [train_list[idx] for idx in order]

        '''generate train_set and test_set using PathsDataset'''
        '''TBD: use TensorDataset if pre-loading in memory'''
        train_set = PathsDataset(
            root=img_folder_path,
            files=train_list,
            transform=transforms.Compose([transforms.Resize(_image_size)]))
        val_set = PathsDataset(
            root=img_folder_path,
            files=val_list,
            transform=transforms.Compose([transforms.Resize(_image_size)]))
        test_set = PathsDataset(
            root=img_folder_path,
            files=test_list,
            transform=transforms.Compose([transforms.Resize(_image_size)]))
        # train_set = AvalancheDataset(PathsDataset(
        #     root=img_folder_path,
        #     files=train_list,
        #     transform=transforms.Resize(_image_size)),
        #     transform_groups={'val': (None, None)})  # allow reshape but not equal scaling
        # val_set = AvalancheDataset(PathsDataset(
        #     root=img_folder_path,
        #     files=val_list,
        #     transform=transforms.Resize(_image_size)),
        #     transform_groups={'val': (None, None)})
        # test_set = AvalancheDataset(PathsDataset(
        #     root=img_folder_path,
        #     files=test_list,
        #     transform=transforms.Resize(_image_size)),
        #     transform_groups={'val': (None, None)})

        datasets = {'train': train_set, 'val': val_set, 'test': test_set}
        label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    elif mode in ['sys', 'pro', 'non', 'noc']:      # no sub
        json_name = {'sys': 'sys_fewshot.json', 'pro': 'pro_fewshot.json',      # 'sub': 'sub_fewshot.json',
                     'non': 'non_fewshot.json', 'noc': 'noc_fewshot.json'}[mode]
        json_path = os.path.join(img_folder_path, json_name)
        with open(json_path, 'r') as f:
            img_info = json.load(f)
        label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
        map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
        map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
        preprocess_label_to_integer(img_info, map_tuple_label_to_int)
        img_list = formulate_img_tuples(img_info)
        dataset = PathsDataset(
            root=img_folder_path,
            files=img_list,
            transform=transforms.Compose([transforms.Resize(_image_size)]))
        # dataset = AvalancheDataset(PathsDataset(
        #     root=img_folder_path,
        #     files=img_list,
        #     transform=transforms.Resize(_image_size)),
        #     transform_groups={'val': (None, None)})

        datasets = {'dataset': dataset}
        label_info = (label_set, map_tuple_label_to_int, map_int_label_to_tuple)

    else:
        raise Exception(f'Un-implemented mode "{mode}".')

    return datasets, label_info


__all__ = ["continual_training_benchmark", "fewshot_testing_benchmark"]


if __name__ == "__main__":
    '''Continual'''
    # _dataset, _label_info = _get_pin_datasets('../../datasets', mode='continual')
    # s = set(np.concatenate([list(item) for item in _label_info[0]]))

    # _benchmark_instance = continual_training_benchmark(
    #     n_experiences=10, return_task_id=True,
    #     seed=1234, shuffle=True,
    #     dataset_root='../../datasets',
    #     # memory_size=1000,
    # )
    # fixed_class_order:

    '''Sys'''
    _dataset, _label_info = _get_pin_datasets('../../datasets', mode='noc')
    s = set(np.concatenate([list(item) for item in _label_info[0]]))

    # _benchmark_instance = fewshot_testing_benchmark(
    #     n_experiences=5, n_way=10, n_shot=10, n_query=10, mode='noc',
    #     task_offset=10,
    #     seed=1234, dataset_root='../../datasets',
    # )

    '''Sub'''



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

    # check_vision_benchmark(_benchmark_instance, show_without_transforms=True)
