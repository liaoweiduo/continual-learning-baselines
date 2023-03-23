from typing import Optional, Sequence, List, Union

import torch
from avalanche.benchmarks import CLExperience
from avalanche.evaluation.metric_definitions import TResult
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.evaluation.metric_results import (
    MetricValue,
    MetricResult,
)

from models.module_net import ModuleNet
from strategies.hsic import hsic_normalized


class SelectModuleMetric(PluginMetric):
    """Metric used to output selected modules on all layers.

    This metric is activated only at eval time.
    """

    def __init__(self):
        super().__init__()
        self.mode = 'eval'
        self.save_image = False
        self._select_matrix = []
        self._labels = []

    def _package_result(self, strategy: "SupervisedTemplate") -> "MetricResult":
        _select_matrix = self.result()

        metric_name = get_metric_name(
            self,
            strategy,
            add_experience=self.mode == "eval",
            add_task=True,
        )

        plot_x_position = strategy.clock.train_iterations

        metric_representation = MetricValue(
            self, metric_name, _select_matrix, plot_x_position
        )

        return [metric_representation]

    def reset(self) -> None:
        self._select_matrix = []
        self._labels = []

    def result(self, **kwargs) -> torch.Tensor:
        if not self._select_matrix:
            return torch.zeros((0, 0), dtype=torch.long)
        _select_matrix = torch.cat(self._select_matrix).cpu()  # [n_samp, 32]

        _labels = torch.cat(self._labels).cpu()     # no use for debug

        return _select_matrix

    def before_eval(self, strategy) -> None:
        self.reset()

    def after_eval(self, strategy: "SupervisedTemplate") -> "MetricResult":
        return self._package_result(strategy)

    def after_eval_iteration(self, strategy: "SupervisedTemplate"):
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        super().after_training_iteration(strategy)

        selected_idxs_each_layer = strategy.model.backbone.selected_idxs_each_layer
        # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]
        selected_idxs_each_layer = torch.stack(selected_idxs_each_layer, dim=1).detach().clone()     # [64, 4, 8]
        bs, n_layers, n_modules = selected_idxs_each_layer.shape
        selected_idxs = selected_idxs_each_layer.reshape(bs, -1)
        dim = n_layers * n_modules
        # [bs, dim]: [64, 32]

        batch_labels = strategy.mbatch[1]  # [bs,]: [64]

        self._select_matrix.append(selected_idxs)
        self._labels.append(batch_labels)

    def __str__(self):
        return "SelectModule"


class SparseSelectionPlugin(SupervisedPlugin):
    """ The selection of modules in each layer should be sparse.
    """

    def __init__(self, alpha=1., activate=True):
        """
        :param alpha: regular coefficient.
        :param activate: whether is activated or not.
        """
        super().__init__()
        self.alpha = alpha
        self.activate = activate
        self.structure_loss = 0

    def after_forward(self, strategy, **kwargs):
        pass

    def before_backward(self, strategy, **kwargs):
        """
        Add mean loss
        """
        if self.activate:
            selected_idxs_each_layer = strategy.model.backbone.selected_idxs_each_layer
            # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]

            structure_loss = torch.mean(torch.stack(selected_idxs_each_layer))
            strategy.loss += self.alpha * structure_loss

            self.structure_loss = structure_loss.item()


class IndependentSelectionPlugin(SupervisedPlugin):
    """ The selection of modules in each layer should be independent for different labels.
    Note that independent does not mean different.
    """

    def __init__(self, alpha=1., activate=True):
        """
        :param alpha: regular coefficient.
        :param activate: whether is activated or not.
        """
        super().__init__()
        self.alpha = alpha
        self.activate = activate
        self.structure_loss = 0

    def before_backward(self, strategy, **kwargs):
        """
        Add HSIC loss
        """
        if self.activate:
            selected_idxs_each_layer = strategy.model.backbone.selected_idxs_each_layer
            # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]
            selected_idxs_each_layer = torch.stack(selected_idxs_each_layer, dim=1)     # [64, 4, 8]
            bs, n_layers, n_modules = selected_idxs_each_layer.shape
            selected_idxs = selected_idxs_each_layer.reshape(bs, -1)
            dim = n_layers * n_modules
            # [bs, dim]: [64, 32]

            '''labels for this batch'''
            batch_labels = strategy.mbatch[1]       # [bs,]: [64]
            label_set = sorted(set(batch_labels.tolist()))
            if len(label_set) == 1:
                # only one class in this mini batch, do not need to cal independent loss
                return

            prototypes = []     # [n_class, 32]
            for label in label_set:
                prototype = torch.mean(selected_idxs[batch_labels == label], dim=0)    # [32]
                prototypes.append(prototype)

            prototypes = torch.stack(prototypes)
            pair_comb_protos = prototypes[torch.combinations(torch.arange(prototypes.shape[0]))]
            # [n_comb, 2, 32]
            A, B = pair_comb_protos[:, 0, :], pair_comb_protos[:, 1, :]

            structure_loss = hsic_normalized(A, B)
            strategy.loss += self.alpha * structure_loss

            self.structure_loss = structure_loss.item()


class ConsistentSelectionPlugin(SupervisedPlugin):
    """ The selection of modules in each layer should be consistent for the same label.
    """

    def __init__(self, alpha=1., activate=True):
        """
        :param alpha: regular coefficient.
        :param activate: whether is activated or not.
        """
        super().__init__()
        self.alpha = alpha
        self.activate = activate
        self.structure_loss = 0

    def before_backward(self, strategy, **kwargs):
        """
        Add distance loss, dot product
        """
        if self.activate:
            selected_idxs_each_layer = strategy.model.backbone.selected_idxs_each_layer
            # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]
            selected_idxs_each_layer = torch.stack(selected_idxs_each_layer, dim=1)     # [64, 4, 8]
            bs, n_layers, n_modules = selected_idxs_each_layer.shape
            selected_idxs = selected_idxs_each_layer.reshape(bs, -1)
            dim = n_layers * n_modules
            # [bs, dim]: [64, 32]

            '''labels for this batch'''
            structure_loss = []
            batch_labels = strategy.mbatch[1]       # [bs,]: [64]
            label_set = sorted(set(batch_labels.tolist()))
            for label in label_set:
                A = selected_idxs[batch_labels == label]
                sim_matrix = A @ A.T
                # mask diag     since increase diag elements (self dot prod) means encourage all 1 (all select).
                sim_matrix = sim_matrix - torch.diag_embed(torch.diag(sim_matrix))

                structure_loss.append(torch.mean(sim_matrix))

            structure_loss = torch.sum(torch.stack(structure_loss))
            structure_loss = - structure_loss       # minimize negative similarity.
            strategy.loss += self.alpha * structure_loss

            self.structure_loss = structure_loss.item()


class Algorithm(SupervisedTemplate):
    """ Select
    """
    def __init__(
        self,
        model: ModuleNet,
        optimizer: Optimizer,
        criterion,
        ssc: Union[float, Sequence[float]],
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
        ssp = SparseSelectionPlugin(ssc)
        isp = IndependentSelectionPlugin(isc)
        csp = ConsistentSelectionPlugin(csc)

        if plugins is None:
            plugins = [rp, ssp, isp, csp]
        else:
            plugins.append(ssp)
            plugins.append(isp)
            plugins.append(csp)

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


__all__ = ['SparseSelectionPlugin', 'IndependentSelectionPlugin', 'ConsistentSelectionPlugin', 'Algorithm']


if __name__ == '__main__':
    pass