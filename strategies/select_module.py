from typing import Optional, Sequence, List, Union

import torch
from avalanche.benchmarks import CLExperience
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

from models.module_net import ModuleNet


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

    def before_backward(self, strategy, **kwargs):
        """
        Add mean loss
        """
        if self.activate:
            selected_idxs_each_layer = strategy.model.selected_idxs_each_layer
            # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]

            structure_loss = torch.mean(torch.stack(selected_idxs_each_layer))
            strategy.loss += self.alpha * structure_loss


class IndependentSelectionPlugin(SupervisedPlugin):
    """ The selection of modules in each layer should be independent for different labels.
    """

    def __init__(self, alpha=1., activate=True):
        """
        :param alpha: regular coefficient.
        :param activate: whether is activated or not.
        """
        super().__init__()
        self.alpha = alpha
        self.activate = activate

    def before_backward(self, strategy, **kwargs):
        """
        Add HSIC loss
        """
        # if self.activate:
        pass


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

    def before_backward(self, strategy, **kwargs):
        """
        Add distance loss
        """
        if self.activate:
            selected_idxs_each_layer = strategy.model.selected_idxs_each_layer
            # [n_layers* tensor[bs, n_modules]]: [4* tensor[64, 8]]

            '''labels for this batch'''
            batch_labels = strategy.mbatch[1]

            # structure_loss =


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
        ssp = SparseSelectionPlugin(ssc)
        isp = IndependentSelectionPlugin(isc)
        csp = ConsistentSelectionPlugin(csc)

        if plugins is None:
            plugins = [ssp, isp, csp]
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