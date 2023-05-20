from typing import Optional, Sequence, List, Union, Dict, Tuple
from collections import Counter
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torchvision import transforms
from torchvision.utils import make_grid

from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

from avalanche.benchmarks.utils import make_classification_dataset

from avalanche.evaluation import PluginMetric, Metric
from avalanche.evaluation.metrics.images_samples import ImagesSamplePlugin
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metric_results import (
    MetricValue,
    LoggingType,
    TensorImage,
    MetricResult,
)


class CAMPluginMetric(ImagesSamplePlugin):
    """Metric used to output CAM visualization.

    Visualizing with images in test stream.

    This plugin is not recommended during training,
    since it load and forward extra images to the model.
    """
    def __init__(self, image_size, benchmark, wandb_log=False,
                 num_samples=1, target=0, stream='test', log_label_str=True):
        """
        :param image_size: 128
        :param benchmark: target benchmark to get samples.
        :param wandb_log: True will upload image and mask to wand.
        :param num_samples: num_samples for each class in the specific task specified as ``target''.
        :param target: target task index for benchmark's test stream to sample images. default 0.
                        note that this is not task_id, since for fewshot test, it usually has task_offset.
        :param stream: train or test.
        :param log_label_str: True if also log label string.
                                Note that images are sorted based on original int labels
                                (which is based on label strings).
        """
        self.original_classes = benchmark.original_classes_in_exp[target]
        self.related_classes = benchmark.classes_in_exp[target]
        self.label_info = benchmark.label_info
        super().__init__(mode="eval", n_cols=num_samples, n_rows=len(self.original_classes))
        self.image_size = image_size
        self.num_samples = num_samples
        self.benchmark = benchmark
        self.wandb_log = wandb_log
        self.target = target
        self.stream = stream
        self.log_label_str = log_label_str
        self.seed = 1234        # for selecting samples for each class

        self.wandb = None
        if wandb_log:
            import wandb
            self.wandb = wandb

        self.eval_transform_no_norm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.classes_in_exp = self.benchmark.classes_in_exp[self.target]        # [10]
        self.original_classes_in_exp = self.benchmark.original_classes_in_exp[self.target]      # [10]
        if isinstance(self.classes_in_exp, np.ndarray):
            self.classes_in_exp: list = self.classes_in_exp.tolist()
        if isinstance(self.original_classes_in_exp, np.ndarray):
            self.original_classes_in_exp: list = self.original_classes_in_exp.tolist()

        self.images, self.images_no_norm, self.labels, self.original_labels, self.tasks = [], [], [], [], []

    def result(self, **kwargs):
        return self.images, self.images_no_norm, self.labels, self.original_labels, self.tasks

    def reset(self):
        self.images, self.images_no_norm, self.labels, self.original_labels, self.tasks = [], [], [], [], []

    def get_weights(
        self, strategy: "SupervisedTemplate"
    ):
        """Get classifier's weights

        If in class-IL, change key of weights from classifiers.classifier.weight to classifiers.0.classifier.weight
        to support task_id selection.
        """
        model = strategy.model
        state_dict = model.classifier.state_dict()
        # task-IL
        # 'classifiers.0.classifier.weight' = {Tensor: (10, 512)}
        # 'classifiers.0.classifier.bias' = {Tensor: (10,)}
        # class-IL
        # 'classifiers.classifier.weight' = {Tensor: (10, 512)}
        # 'classifiers.classifier.bias' = {Tensor: (10,)}

        # to support task_id selection for class-IL
        if 'classifiers.classifier.weight' in state_dict.keys():
            state_dict['classifiers.0.classifier.weight'] = state_dict['classifiers.classifier.weight']

        weights = {}
        for key, params in state_dict.items():
            if key.startswith('classifiers') and key.endswith('weight') and len(key.split('.')) == 4:
                # classifiers.{task_id}.classifier.weight
                task_id = int(key.split('.')[1])
                assert task_id not in weights.keys()
                weights[task_id] = params

        return weights

    def _normalize(self, cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams

    def overlay_mask(self,
                     img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
        """Overlay a colormapped mask on a background image

        Args:
            img: background image
            mask: mask to be overlayed in grayscale
            colormap: colormap to be applied on the mask
            alpha: transparency of the background image

        Returns:
            overlayed image
        """

        if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
            raise TypeError('img and mask arguments need to be PIL.Image')

        if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
            raise ValueError('alpha argument is expected to be of type float between 0 and 1')

        cmap = cm.get_cmap(colormap)
        # Resize mask and apply colormap
        overlay = mask.resize(img.size, resample=Image.BICUBIC)
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
        # Overlay the image with the mask
        overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

        return overlayed_img

    def apply_cam(self, feature, weight_softmax, class_idx):
        """
        :param feature: [512, hl, wl]
        :param weight_softmax: [nc, 512]
        :param class_idx: from 0 to nc-1
        """
        d, h, w = feature.shape
        nc, d2 = weight_softmax.shape
        assert d == d2
        cam = (weight_softmax[class_idx].view(d, 1, 1) * feature).sum(0)        # dot product [hl, wl]
        cam = self._normalize(F.relu(cam, inplace=True)).cpu()
        mask = to_pil_image(cam.detach().numpy(), mode='F')
        return mask

    def _make_grid_sample(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        """Modify: add obtain layer_output"""
        self._load_sorted_images(strategy)

        '''Forward model's get_layer_output(x, -1) to get hidden_features'''
        model = strategy.model
        batched_images = torch.stack(self.images).to(strategy.device)       # [bs, 3, H, W]
        batched_images_no_norm = torch.stack(self.images_no_norm)
        with torch.no_grad():
            _, hidden_features = model.resnet.get_layer_output(batched_images, -1)    # [BS, 512, Hl, Wl]

        # another method use hook
        # feature_map = []  # 建立列表容器，用于盛放输出特征图
        #
        # def forward_hook(module, inp, outp):  # 定义hook
        #     feature_map.append(outp)  # 把输出装入字典feature_map
        #
        # model.resnet.layer4.register_forward_hook(forward_hook)  # 对net.layer4这一层注册前向传播
        # with torch.no_grad():
        #     _ = model.resnet(batched_images)  # 前向传播
        #
        # print(feature_map[0].size())

        '''Use task_id to select correct weights'''
        weights = self.get_weights(strategy)    # dict {task_id: [n_classes, 512]}

        '''Precess image one by one because their might be multiple task_ids'''
        masked_images = []
        trans = transforms.ToTensor()
        for hidden_feature, images_no_norm, related_label, task_id in zip(
                hidden_features, batched_images_no_norm, self.labels, self.tasks):
            mask = self.apply_cam(hidden_feature, weights[task_id], related_label)      # PIL.Image
            orign_img = to_pil_image(images_no_norm, mode='RGB')
            masked_images.append(trans(self.overlay_mask(orign_img, mask)))       # tensor

        n_col = self.n_cols
        grid_img = make_grid(masked_images, normalize=False, nrow=n_col)

        '''class string'''
        label_text = None
        if self.log_label_str:
            label_collect = []
            for label in self.original_labels:
                if label not in label_collect:
                    label_collect.append(label)
            label_text = [[str(self.benchmark.label_info[2][label])] for label in label_collect]
            # [['label 1'], ['label 2']]

        metric_name = get_metric_name(
                    self,
                    strategy,
                    add_experience=self.mode == "eval",
                    add_task=True,
                )

        if self.wandb_log:
            log_dict = {metric_name: self.wandb.Image(grid_img)}
            self.wandb.log(log_dict)
            if label_text is not None:
                log_dict = {f'{metric_name}/label_string': self.wandb.Table(
                    columns=["label"], data=label_text)}
                self.wandb.log(log_dict)

        self.reset()

        return_metrics = [
            MetricValue(
                self,
                name=metric_name,
                value=grid_img.numpy(),
                x_plot=strategy.clock.train_iterations,
            ),
        ]

        return return_metrics

    def _load_sorted_images(self, strategy: "SupervisedTemplate"):
        """Modify: labels and tasks as instance variables"""
        self.reset()
        self.images, self.images_no_norm, self.labels, self.original_labels, self.tasks = self._load_data(strategy)
        if self.group:
            self._sort_images_with_labels()

    def _load_data(
        self, strategy: "SupervisedTemplate"
    ) -> Tuple[List[Tensor], List[Tensor], List[int], List[int], List[int]]:
        """Modify: apply norm transform on images. provide w/ and w/o norm versions of images"""
        stream = self.benchmark.test_stream if self.stream == 'test' else self.benchmark.train_stream
        dataset = stream[self.target].dataset.replace_current_transform_group(
            self.eval_transform_no_norm)

        images, images_no_norm, labels, original_labels, tasks = [], [], [], [], []
        task_label_image_dict = {}

        '''collect images in the dataset'''
        for item in dataset:
            image, label, task = item
            if not isinstance(label, int):
                label = label.item()
            if not isinstance(task, int):
                task = task.item()
            if task not in task_label_image_dict.keys():
                task_label_image_dict[task] = {}
            if label not in task_label_image_dict[task].keys():
                task_label_image_dict[task][label] = []

            task_label_image_dict[task][label].append(image)

        '''sample self.num_samples images for each label-task pair'''
        # torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        for task in task_label_image_dict.keys():
            for label in task_label_image_dict[task].keys():
                image_list = task_label_image_dict[task][label]
                selected_idxs = rng.permutation(np.arange(len(image_list)))[:self.num_samples]
                # selected_idxs = rng.choice(np.arange(len(image_list)), self.num_samples)
                for idx in selected_idxs:
                    images_no_norm.append(image_list[idx])
                    images.append(self.norm_transform(image_list[idx]))
                    labels.append(label)
                    tasks.append(task)

                    label_index = self.classes_in_exp.index(label)
                    original_labels.append(self.original_classes_in_exp[label_index])

        return images, images_no_norm, labels, original_labels, tasks

    def _sort_images_with_labels(self):
        """Modify: also change self.labels and self.tasks and image_no_norm"""
        ts, ls, ols, ims, imns = [], [], [], [], []
        for task, label, origin_label, image, image_no_norm in sorted(
                zip(self.tasks, self.labels, self.original_labels, self.images, self.images_no_norm),
                key=lambda t: (t[0], t[1]),     # sort by task and label
        ):
            ts.append(task)
            ls.append(label)
            ols.append(origin_label)
            ims.append(image)
            imns.append(image_no_norm)
        self.images = ims
        self.images_no_norm = imns
        self.labels = ls
        self.original_labels = ols
        self.tasks = ts

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        """do nothing before training"""
        pass

    def after_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        """do nothing before evaluating"""
        pass

    def after_training(self, strategy: "SupervisedTemplate") -> "MetricResult":
        """Do CAM visualization after learning the classifier"""
        return self._make_grid_sample(strategy)

    def __str__(self):
        return "CAM"
