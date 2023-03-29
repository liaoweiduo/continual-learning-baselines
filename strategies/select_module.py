from typing import Optional, Sequence, List, Union, Dict
from collections import Counter

import torch
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
        bs, n_layers, n_modules = selected_idxs_each_layer.shape
        selected_idxs = selected_idxs_each_layer.reshape(bs, -1)
        dim = n_layers * n_modules
        # [bs, dim]: [64, 32]

        '''labels for this batch'''
        batch_labels = strategy.mbatch[1]       # [bs,]: [64]

        self._select_matrix.append(selected_idxs)
        self._labels.append(batch_labels)

    def result(
            self, matrix=True, sparse=True, supcon=True,
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
            dic['Matrix'] = select_matrix.detach().cpu() if return_float else select_matrix
            dic['Labels'] = labels.detach().cpu() if return_float else labels

        if sparse:
            sparse_loss = self.get_sparse_selection_loss()
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

        return dic

    def reset(self):
        self._select_matrix = []
        self._labels = []

    def prepare_matrix(self):
        """cat matrix and labels"""
        if len(self._select_matrix) == 0:
            return None, None

        select_matrix = torch.cat(self._select_matrix)  # [n_samp, 32]
        labels = torch.cat(self._labels)
        return select_matrix, labels

    def get_sparse_selection_loss(self):
        """l1 norm: torch.mean"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        structure_loss = torch.mean(select_matrix)

        return structure_loss

    def get_sup_con_loss(self):
        """Super Contrastive Loss"""
        select_matrix, labels = self.prepare_matrix()
        if select_matrix is None:
            return torch.tensor(0)

        # if any of label exists only once, sup con fails and will output nan, instead, we output 0
        if min(Counter(labels.tolist()).values()) == 1:
            return torch.tensor(0)

        'norm to Euclidean dist = 1'
        select_matrix = select_matrix.unsqueeze(1)    # [n_samp, 1, 32]
        select_matrix = select_matrix / torch.norm(select_matrix, dim=2, p=2, keepdim=True)

        criterion = SupConLoss()
        structure_loss = criterion(select_matrix, labels)

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

    def __init__(self, mode='both', matrix=True, sparse=True, supcon=True, independent=False, consistent=False):
        super().__init__()
        assert (mode in ['both', 'train', 'eval']
                ), f'Current mode is {mode}, should be one of [both, train, eval].'
        self.mode = mode
        self.save_image = False     # todo: need to implement
        self.matrix = matrix
        self.sparse = sparse
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
        dic = self._metric.result(matrix=self.matrix, sparse=self.sparse, supcon=self.supcon,
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
            return self._package_result(strategy)

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
            return self._package_result(strategy)

    def __str__(self):
        return "SelectModule"


class ImageSimilarityPluginMetric(ImagesSamplePlugin):
    """Metric used to output similarity tensor.

    Visualizing with images.

    This plugin is not recommended during training,
    since it load and forward extra images to the model.
    """
    def __init__(self, active=False, image_size=128, num_samples=100, num_proto=28):
        super().__init__(mode="eval", n_cols=1, n_rows=num_samples)
        self.active = active
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_proto = num_proto      # 4*7

        self.eval_transform_no_norm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])      # no use

        self.images = []
        self.labels = []
        self.tasks = []
        self.similarity_tensors = []

    def set_active(self, active: bool):
        self.active = active

    def result(self, **kwargs):
        return self.images, self.labels, self.tasks, self.similarity_tensors

    def reset(self):
        self.images, self.labels, self.tasks, self.similarity_tensors = [], [], [], []

    def _make_grid_sample(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        """Modify: add obtain similarity_tensors"""
        self._load_sorted_images(strategy)

        '''Forward model to get similarity_tensors'''
        model = strategy.model
        batched_images = torch.stack(self.images).to(model.classifier.active_units_T0.device)       # [bs, 3, H, W]

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
            bs, n_proto, Hl, Wl = similarity_tensor_each_layer.shape
            c3tensor = torch.stack(
                [similarity_tensor_each_layer, similarity_tensor_each_layer, similarity_tensor_each_layer],
                dim=2
            ).reshape(-1, 3, Hl, Wl)       # [bs*n_proto, 3, Hl, Wl]
            resized_mask = torch.stack([resize_trans(img) for img in c3tensor])
            H, W = resized_mask.shape[-2:]
            resized_mask = resized_mask.reshape(bs, n_proto, 3, H, W)
            masks.append(resized_mask)
        masks = torch.cat(masks, dim=1)     # [bs, n_layer*n_proto, 3, H, W]

        images = torch.cat([batched_images.unsqueeze(1), masks], dim=1)     # [bs, 1+n_layer*n_proto, 3, H, W]
        images = images.reshape(-1, *images.shape[-3:])      # [bs*(1+n_layer*n_proto), 3, H, W]

        return [
            MetricValue(
                self,
                name=get_metric_name(
                    self,
                    strategy,
                    add_experience=self.mode == "eval",
                    add_task=True,
                ),
                value=TensorImage(
                    make_grid(
                        list(images), normalize=False, nrow=self.num_proto+1
                    )
                ),
                x_plot=strategy.clock.train_iterations,
            )
        ]

    def _load_sorted_images(self, strategy: "SupervisedTemplate"):
        """Modify: labels and tasks as instance variables"""
        self.reset()
        self.images, self.labels, self.tasks = self._load_data(strategy)
        if self.group:
            self._sort_images(self.labels, self.tasks)

    def _sort_images(self, labels: List[int], tasks: List[int]):
        """Modify: also change self.labels and self.tasks"""
        ts, ls, ims = [], [], []
        for task, label, image in sorted(
                zip(tasks, labels, self.images),
                key=lambda t: (t[0], t[1]),
        ):
            ts.append(task)
            ls.append(label)
            ims.append(image)
        self.images = ims
        self.labels = ls
        self.tasks = ts

    def _make_dataloader(
        self, data: "make_classification_dataset", mb_size: int
    ) -> DataLoader:
        """Modify: use its transform, since we need forward images"""
        # todo: if need to apply similarity mask on samples without normalization
        # data_no_norm = data.replace_current_transform_group(self.eval_transform_no_norm)
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
    def __init__(self, sparse_alpha=1., supcon_alpha=1., independent_alpha=0., consistent_alpha=0.):
        """
        :param sparse_alpha: sparse reg coefficient.
        :param supcon_alpha: supcon reg coefficient.
        :param independent_alpha: independent reg coefficient.
        :param consistent_alpha: consistent reg coefficient.
        """
        super().__init__()
        self.sparse_alpha = sparse_alpha
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
            structure_loss = structure_loss + self.sparse_alpha * self._metric.get_sparse_selection_loss()
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
        sp = SelectionPlugin(sparse_alpha=ssc, supcon_alpha=scc, independent_alpha=isc, consistent_alpha=csc)

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
