from typing import Optional, Sequence, List, Union, Dict, Tuple
from collections import Counter
import math

import torch
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

from models.module_net import ModuleNet
from strategies.hsic import hsic_normalized
from strategies.sup_con_loss import SupConLoss


class SelectionMetric(Metric):
    """The selection of modules in each layer.

    The selection is represented by a tensor with shape [num_samples, num_layers*num_modules].
    Also calculate three regularizers including:
        Sparse selection loss:
            The selection of modules in each layer should be sparse.
        Independent selection loss:
            The selection of modules in each layer should be independent for different labels.
            Note that independent does not mean different.
        Consistent selection loss:
            The selection of modules in each layer should be consistent for the same label.
    """

    def __init__(self, save_image=False):
        # todo: implement save_image=True
        self.save_image = save_image

        self._select_matrix = []
        self._similarity_tensors = []
        self._labels = []

    def update(self, strategy):
        """ Update the samples.

        Should not use @torch.no_grad(), since this Metric is also used for computing regs in training.
        :param strategy: the model in this strategy should be ModuleNet.
        :param selected_idxs: tensor with shape [bs, dim], dim is num_layers*num_modules by default.
        :param batch_labels: long tensor with shape [bs], labels.
        """
        selected_idxs_each_layer = strategy.model.backbone.selected_idxs_each_layer
        # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]
        selected_idxs_each_layer = torch.stack(selected_idxs_each_layer, dim=1)     # [64, 4, 8]
        similarity_tensor = strategy.model.backbone.similarity_tensor
        # n_layer*[bs, n_proto, Hl, Wl]     since they have different shape in each layer, can not be stack
        bs, n_layers, n_modules = selected_idxs_each_layer.shape
        selected_idxs = selected_idxs_each_layer.reshape(bs, -1)
        self.n_layers = n_layers
        self.n_modules = n_modules
        dim = n_layers * n_modules
        # [bs, dim]: [64, 32]

        '''labels for this batch'''
        batch_labels = strategy.mbatch[1]       # [bs,]: [64]

        self._select_matrix.append(selected_idxs)
        self._similarity_tensors.append(similarity_tensor)
        self._labels.append(batch_labels)

    def result(
            self, matrix=True, simi=False, sparse=True, sparse_threshold=2,
            supcon=True,
            independent=False, consistent=False,
            return_float=True, **kwargs
    ):
        """Cat select_matrix and labels for output
        :param matrix: whether to output matrix
        :param sparse: whether to output sparse selection loss
        :param supcon: whether to output SupCon loss
        :param independent: whether to output independent selection loss
        :param consistent: whether to output consistent selection loss
        :param return_float: whether to return float for scalars and detach and cpu for Tensors.
        """
        dic = {}
        if matrix:
            select_matrix, labels = self.prepare_matrix()
            dic['Matrix'] = select_matrix.detach().cpu().numpy() if return_float else select_matrix
            dic['Labels'] = labels.detach().cpu().numpy() if return_float else labels

            if simi:
                # similarity tensors
                similarity_tensors = self._similarity_tensors       # [n_batch* [n_layers* [bs, n_proto, Hl, Wl]]]
                for layer_idx in range(len(similarity_tensors[0])):
                    for proto_idx in range(similarity_tensors[0][0].shape[1]):
                        per_layer_per_proto = torch.cat(
                            [similarity_tensors[b_idx][layer_idx][:, proto_idx]
                             for b_idx in range(len(similarity_tensors))])
                        dic[f'SimilarityMatrix.l{layer_idx}.p{proto_idx}'] = per_layer_per_proto.detach(
                        ).cpu().numpy() if return_float else per_layer_per_proto

        if sparse:
            sparse_loss = self.get_sparse_selection_loss(sparse_threshold)
            dic['Sparse'] = sparse_loss.item() if return_float else sparse_loss

        if supcon:
            supcon_loss = self.get_sup_con_loss()
            dic['SupCon'] = supcon_loss.item() if return_float else supcon_loss

        if independent:
            independent_loss = self.get_independent_selection_loss()
            dic['Independent'] = independent_loss.item() if return_float else independent_loss

        if consistent:
            consistent_loss = self.get_consistent_selection_loss()
            dic['Consistent'] = consistent_loss.item() if return_float else consistent_loss

        self.reset()

        return dic

    def reset(self):
        self._select_matrix = []
        self._similarity_tensors = []
        self._labels = []

    def prepare_matrix(self):
        """cat matrix and labels"""
        if len(self._select_matrix) == 0:
            return None, None

        select_matrix = torch.cat(self._select_matrix)  # [n_samp, 32]
        labels = torch.cat(self._labels)
        return select_matrix, labels

    def get_sparse_selection_loss(self, sparse_threshold=2):
        """l1 norm: torch.mean"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        bs = select_matrix.shape[0]
        n_layers = self.n_layers
        n_modules = self.n_modules
        assert select_matrix.shape[1] == n_layers * n_modules

        # penalty num_module selected larger than sparse_threshold
        select_matrix = select_matrix.reshape(bs, n_layers, n_modules)
        sum_over_layer = torch.sum(select_matrix, dim=-1)
        # mask out <= sparse_threshold+1, here +1 for identity module
        # print(f'sum_over_layer: {sum_over_layer}: {sum_over_layer.shape}, {sum_over_layer.dtype}. ')
        sum_over_layer = torch.where(sum_over_layer > sparse_threshold + 1,
                                     sum_over_layer, torch.zeros_like(sum_over_layer))

        structure_loss = torch.sum(sum_over_layer) / (bs * n_layers * n_modules)

        # penalty mask for identity modules
        # mask = torch.ones_like(select_matrix)
        # for layer_idx in range(n_layers):
        #     mask[:, (layer_idx + 1) * n_modules - 1] = 10
        #
        # structure_loss = torch.mean(select_matrix * mask)

        return structure_loss

    def get_sup_con_loss(self):
        """Super Contrastive Loss"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        # if any of label exists only once, sup con fails and will output nan, instead, we output 0
        if min(Counter(labels.tolist()).values()) == 1:
            return torch.tensor(0)

        '''mask out identity'''
        bs = select_matrix.shape[0]
        n_layers = self.n_layers
        n_modules = self.n_modules
        assert select_matrix.shape[1] == n_layers * n_modules
        select_matrix = select_matrix.reshape(bs, n_layers, n_modules)
        select_matrix = select_matrix[:, :, :-1]        # [bs, n_layers, n_modules - 1]
        select_matrix = select_matrix.reshape(bs, n_layers * (n_modules - 1))

        'norm to Euclidean dist = 1'
        select_matrix = select_matrix.unsqueeze(1)    # [n_samp, 1, 28]
        select_matrix = select_matrix / (torch.norm(select_matrix, dim=2, p=2, keepdim=True) + 1e-15)
        # all 0 will have nan

        criterion = SupConLoss()
        structure_loss = criterion(select_matrix, labels)

        if torch.isnan(structure_loss):
            print(f'select_matrix: {select_matrix.tolist()}')
            print(f'labels: {labels}')

            raise Exception('structure_loss is nan')

        return structure_loss

    def get_independent_selection_loss(self):
        """HSIC loss over all combinations of class-pairs"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        label_set = sorted(set(labels.tolist()))
        if len(label_set) == 1:
            # only one class in this mini batch, do not need to cal independent loss
            return 0

        prototypes = []     # [n_class, 32]
        for label in label_set:
            prototype = torch.mean(select_matrix[labels == label], dim=0)    # [32]
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)
        pair_comb_protos = prototypes[torch.combinations(torch.arange(prototypes.shape[0]))]
        # [n_comb, 2, 32]
        A, B = pair_comb_protos[:, 0, :], pair_comb_protos[:, 1, :]

        structure_loss = hsic_normalized(A, B)

        return structure_loss

    def get_consistent_selection_loss(self):
        """pair-wise dot product of samples in the same class"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        label_set = sorted(set(labels.tolist()))
        structure_loss = []
        for label in label_set:
            A = select_matrix[labels == label]
            sim_matrix = A @ A.T
            # mask diag     since increase diag elements (self dot prod) means encourage all 1 (all select).
            sim_matrix = sim_matrix - torch.diag_embed(torch.diag(sim_matrix))

            structure_loss.append(torch.mean(sim_matrix))

        structure_loss = - torch.mean(torch.stack(structure_loss)) / 10       # /10 for its large scale
        # increase similarity -> minimize negative similarity.

        return structure_loss


class SelectionPluginMetric(PluginMetric):
    """Metric used to output selected modules on all layers.
    """

    def __init__(self, mode='both', matrix=True, simi=False,
                 sparse=True, sparse_threshold=2,
                 supcon=True, independent=False, consistent=False):
        super().__init__()
        assert (mode in ['both', 'train', 'eval']
                ), f'Current mode is {mode}, should be one of [both, train, eval].'
        self.mode = mode
        self.save_image = False
        self.matrix = matrix
        self.simi = simi
        self.sparse = sparse
        self.sparse_threshold = sparse_threshold
        self.supcon = supcon
        self.independent = independent
        self.consistent = consistent

        self._metric = SelectionMetric(self.save_image)

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        dic = self.result()

        metrics = []
        for k, v in dic.items():
            metric_name = get_metric_name(
                self,
                strategy,
                add_experience=True,        # self.mode in ["eval", "both"],
                add_task=True,
            ) + f'/{k}'
            plot_x_position = strategy.clock.train_iterations

            metric_representation = MetricValue(self, metric_name, v, plot_x_position)

            metrics.append(metric_representation)

        return metrics

    def reset(self) -> None:
        self._metric = SelectionMetric(self.save_image)

    def result(self, **kwargs):
        dic = self._metric.result(matrix=self.matrix, simi=self.simi,
                                  sparse=self.sparse, sparse_threshold=self.sparse_threshold,
                                  supcon=self.supcon,
                                  independent=self.independent, consistent=self.consistent)

        return dic

    def update(self, strategy, **kwargs):
        self._metric.update(strategy)

    def before_training_iteration(self, strategy: "SupervisedTemplate"):
        super().before_training_iteration(strategy)
        if self.mode in ["train", "both"]:
            self.reset()

    def after_training_iteration(self, strategy: "SupervisedTemplate"):
        super().after_training_iteration(strategy)
        if self.mode in ["train", "both"]:
            self.update(strategy)
            result = self._package_result(strategy)
            self.reset()
            return result

    def before_eval_exp(self, strategy: "SupervisedTemplate"):
        super().before_eval_exp(strategy)
        if self.mode in ["eval", "both"]:
            self.reset()

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        super().after_eval_iteration(strategy)
        if self.mode in ["eval", "both"]:
            self.update(strategy)

    def after_eval_exp(self, strategy: "SupervisedTemplate"):
        super().after_eval_exp(strategy)
        if self.mode in ["eval", "both"]:
            result = self._package_result(strategy)
            self.reset()
            return result

    def __str__(self):
        return "SelectModule"


class ImageSimilarityPluginMetric(ImagesSamplePlugin):
    """Metric used to output similarity tensor.

    Visualizing with images.

    This plugin is not recommended during training,
    since it load and forward extra images to the model.
    """
    def __init__(self, image_size, active=False, wandb_log=False, num_samples=10, num_proto=28):
        super().__init__(mode="eval", n_cols=1, n_rows=num_samples)
        self.active = active        # since the eval during training is not needed for collecting masks
        self.wandb_log = wandb_log
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_proto = num_proto      # 4*7

        self.wandb = None
        if wandb_log:
            import wandb
            self.wandb = wandb

        self.eval_transform_no_norm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.images = []
        self.images_no_norm = []
        self.labels = []
        self.tasks = []

    def set_active(self, active: bool):
        self.active = active

    def result(self, **kwargs):
        return self.images, self.images_no_norm, self.labels, self.tasks

    def reset(self):
        self.images, self.images_no_norm, self.labels, self.tasks = [], [], [], []

    def _make_grid_sample(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        """Modify: add obtain similarity_tensors"""
        self._load_sorted_images(strategy)

        '''Forward model to get similarity_tensors'''
        model = strategy.model
        batched_images = torch.stack(self.images).to(strategy.device)       # [bs, 3, H, W]
        batched_images_no_norm = torch.stack(self.images_no_norm)
        c = batched_images_no_norm.shape[-3]

        _ = model.backbone(batched_images)
        similarity_tensor = model.backbone.similarity_tensor    # n_layer*[bs, n_proto, Hl, Wl]

        # reshape mask in each layer to image_size: Hl->H, Wl->W
        resize_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        masks = []
        for layer_id, similarity_tensor_each_layer in enumerate(similarity_tensor):
            bs, n_proto, hl, wl = similarity_tensor_each_layer.shape
            resized_mask = torch.stack([
                resize_trans(img)
                for img in similarity_tensor_each_layer.reshape(-1, hl, wl)
            ])
            h, w = resized_mask.shape[-2:]
            resized_mask = resized_mask.reshape(bs, n_proto, h, w)
            masks.append(resized_mask)
        masks = torch.stack(masks, dim=1)     # [bs, n_layer, n_proto, H, W]    n_layer*n_proto = self.num_proto
        bs, n_layer, n_proto, h, w = masks.shape
        #     c3tensor = torch.stack(
        #         [similarity_tensor_each_layer, similarity_tensor_each_layer, similarity_tensor_each_layer],
        #         dim=2
        #     ).reshape(-1, 3, Hl, Wl)       # [bs*n_proto, 3, Hl, Wl]
        #     resized_mask = torch.stack([resize_trans(img) for img in c3tensor])
        #     H, W = resized_mask.shape[-2:]
        #     resized_mask = resized_mask.reshape(bs, n_proto, 3, H, W)
        #     masks.append(resized_mask)
        # masks = torch.stack(masks, dim=1)     # [bs, n_layer, n_proto, 3, H, W]    n_layer*n_proto = self.num_proto
        # bs, n_layer, n_proto, c, h, w = masks.shape

        metric_name = get_metric_name(
                    self,
                    strategy,
                    add_experience=self.mode == "eval",
                    add_task=True,
                )

        if self.wandb_log:
            # prepare masks for wandb image for each prototype with name: ``l{l}.p{p}''
            n_col = min(bs, 2)
            grid_img = make_grid(list(batched_images_no_norm), normalize=False, nrow=n_col)

            for layer_idx in range(n_layer):
                mask_dict = {}
                for proto_idx in range(n_proto):

                    # binarize mask
                    grid_mask = make_grid(
                        list(masks[:, layer_idx, proto_idx].unsqueeze(1)), normalize=False, nrow=n_col)[0]
                    # [0] for convert [3, HH, WW] to [HH, WW]
                    binarized_grid_mask = (grid_mask > 0.2).int()       # threshold 0.5

                    mask_dict[f'proto{proto_idx}'] = {
                        "mask_data": binarized_grid_mask.numpy(),
                        "class_labels": {0: "0", 1: "1"},
                    }

                wandb_image = self.wandb.Image(grid_img, masks=mask_dict)
                self.wandb.log({f"{metric_name}/l{layer_idx}": wandb_image})

        # apply mask on images
        masks = masks - masks.min()
        masks = masks / masks.max()      # normalize to [0, 1]
        # masks = masks * 0.5 + 0.5     # normalize to [0.5, 1]
        colored_masks = masks.reshape([bs, n_layer, n_proto, 1, h, w])
        colored_masks = torch.cat((colored_masks, 0*colored_masks, 0*colored_masks), dim=-3)    # blue
        masked_images = batched_images_no_norm.reshape(
            [bs, 1, 1, c, h, w]) + 0.5 * colored_masks

        images = torch.cat([batched_images_no_norm.unsqueeze(1),
                            masked_images.reshape([bs, -1, c, h, w])], dim=1)     # [bs, 1+n_layer*n_proto, 3, H, W]
        images = images.reshape(-1, *images.shape[-3:])  # [bs*(1+n_layer*n_proto), 3, H, W]

        self.reset()

        return [
            MetricValue(
                self,
                name=metric_name,
                value=make_grid(
                    list(images), normalize=False, nrow=self.num_proto + 1
                ).numpy(),
                x_plot=strategy.clock.train_iterations,
            ),
            MetricValue(
                self,
                name=f"{metric_name}/images",
                value=make_grid(
                    list(batched_images_no_norm), normalize=False, nrow=min(bs, 5)
                ).numpy(),
                x_plot=strategy.clock.train_iterations,
            ),
            *[MetricValue(
                self,
                name=f"{metric_name}/masks.l{l}.p{p}",
                value=make_grid(
                    list(masks[:, l, p].unsqueeze(1)), normalize=False, nrow=min(bs, 5)
                ).numpy(),
                x_plot=strategy.clock.train_iterations,
            ) for l in range(n_layer) for p in range(n_proto)],
        ]

    def _load_sorted_images(self, strategy: "SupervisedTemplate"):
        """Modify: labels and tasks as instance variables"""
        self.reset()
        self.images, self.images_no_norm, self.labels, self.tasks = self._load_data(strategy)
        if self.group:
            self._sort_images(self.labels, self.tasks)

    def _load_data(
        self, strategy: "SupervisedTemplate"
    ) -> Tuple[List[Tensor], List[Tensor], List[int], List[int]]:
        """Modify: apply norm transform on images. provide w/ and w/o norm versions of images"""
        dataloader = self._make_dataloader(
            strategy.adapted_dataset, strategy.eval_mb_size
        )

        images, images_no_norm, labels, tasks = [], [], [], []

        # todo: num_img for (label, task)
        for batch_images, batch_labels, batch_tasks in dataloader:
            n_missing_images = self.n_wanted_images - len(images)
            labels.extend(batch_labels[:n_missing_images].tolist())
            tasks.extend(batch_tasks[:n_missing_images].tolist())
            raw_images = batch_images[:n_missing_images]
            images_no_norm.extend([
                img for img in raw_images
            ])
            images.extend([
                self.norm_transform(img)
                for img in raw_images
            ])
            if len(images) == self.n_wanted_images:
                return images, images_no_norm, labels, tasks

    def _sort_images(self, labels: List[int], tasks: List[int]):
        """Modify: also change self.labels and self.tasks and image_no_norm"""
        ts, ls, ims, imns = [], [], [], []
        for task, label, image, image_no_norm in sorted(
                zip(tasks, labels, self.images, self.images_no_norm),
                key=lambda t: (t[0], t[1]),
        ):
            ts.append(task)
            ls.append(label)
            ims.append(image)
            imns.append(image_no_norm)
        self.images = ims
        self.images_no_norm = imns
        self.labels = ls
        self.tasks = ts

    def _make_dataloader(
        self, data: "make_classification_dataset", mb_size: int
    ) -> DataLoader:
        """Modify: use transform without norm"""
        data = data.replace_current_transform_group(self.eval_transform_no_norm)
        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
        return DataLoader(
            dataset=data,
            batch_size=min(mb_size, self.n_wanted_images),
            shuffle=True,
            collate_fn=collate_fn,
        )

    def after_train_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.active:
            return super().after_train_dataset_adaptation(strategy)

    def after_eval_dataset_adaptation(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.active:
            return super().after_eval_dataset_adaptation(strategy)

    def __str__(self):
        return "Images_with_Similarity"


class SelectionPlugin(SupervisedPlugin):
    """ Selection Reg Plugin
    Including:
        Sparse selection plugin:
            The selection of modules in each layer should be sparse.
        Independent selection plugin:
            The selection of modules in each layer should be independent for different labels.
            Note that independent does not mean different.
        Consistent selection plugin:
            The selection of modules in each layer should be consistent for the same label.
    """
    def __init__(self, sparse_alpha=1., sparse_threshold=2,
                 supcon_alpha=1., independent_alpha=0., consistent_alpha=0.):
        """
        :param sparse_alpha: sparse reg coefficient.
        :param supcon_alpha: supcon reg coefficient.
        :param independent_alpha: independent reg coefficient.
        :param consistent_alpha: consistent reg coefficient.
        """
        super().__init__()
        self.sparse_alpha = sparse_alpha
        self.sparse_threshold = sparse_threshold
        self.supcon_alpha = supcon_alpha
        self.independent_alpha = independent_alpha
        self.consistent_alpha = consistent_alpha

        self._metric = SelectionMetric(save_image=False)

    def reset(self) -> None:
        self._metric = SelectionMetric(save_image=False)

    def update(self, strategy, **kwargs):
        self._metric.update(strategy)

    def before_backward(self, strategy, **kwargs):
        """
        Apply regs on loss, if the corresponding alpha > 0
        """
        self.reset()
        self.update(strategy)       # collect select matrix for this batch

        structure_loss = 0

        if self.sparse_alpha > 0:
            structure_loss = structure_loss + self.sparse_alpha * self._metric.get_sparse_selection_loss(
                self.sparse_threshold)
        if self.supcon_alpha > 0:
            structure_loss = structure_loss + self.supcon_alpha * self._metric.get_sup_con_loss()
        if self.independent_alpha > 0:
            structure_loss = structure_loss + self.independent_alpha * self._metric.get_independent_selection_loss()
        if self.consistent_alpha > 0:
            structure_loss = structure_loss + self.consistent_alpha * self._metric.get_consistent_selection_loss()

        strategy.loss = strategy.loss + structure_loss


class Algorithm(SupervisedTemplate):
    """ Select
    """
    def __init__(
        self,
        model: ModuleNet,
        optimizer: Optimizer,
        criterion,
        ssc: Union[float, Sequence[float]],
        ssc_threshold: Union[int, Sequence[int]],
        scc: Union[float, Sequence[float]],
        isc: Union[float, Sequence[float]],
        csc: Union[float, Sequence[float]],
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ssc: coefficient for SparseSelection.
        :param scc: coefficient for SupConLoss.
        :param isc: coefficient for IndependentSelection.
        :param csc: coefficient for ConsistentSelection.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        rp = ReplayPlugin(mem_size)
        sp = SelectionPlugin(sparse_alpha=ssc, sparse_threshold=ssc_threshold,
                             supcon_alpha=scc, independent_alpha=isc, consistent_alpha=csc)

        if plugins is None:
            plugins = [rp, sp]
        else:
            plugins.extend([rp, sp])

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


__all__ = ['SelectionMetric', 'SelectionPluginMetric', 'ImageSimilarityPluginMetric', 'SelectionPlugin', 'Algorithm']


if __name__ == '__main__':
    pass
