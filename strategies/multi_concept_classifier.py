from typing import Optional, Sequence, List, Union, Dict, Tuple

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class MultiConceptClassifier(SupervisedPlugin):
    """ Additional multi concept classifier.
    """
    def __init__(self, model, benchmark, weight=1., mask_origin_loss=False):
        """
        :param model: A target model which needs a multi_class_classifier.
        :param benchmark: A benchmark provided label_info for transforming label into multi concept labels.
        :param weight: weight when apply the multi concept loss
        """
        super().__init__()
        self.model = model
        self.benchmark = benchmark
        self.weight = weight
        self.mask_origin_loss = mask_origin_loss

        self.criterion = BCEWithLogitsLoss()       # contain hidden sigmoid

        assert (hasattr(model, "multi_class_classifier")
                ), f'The model should contain a multi_class_classifier.'
        assert (hasattr(benchmark, "label_info")
                ), f'The benchmark should contain a label_info.'

        self.label_set, _, self.map_int_label_to_tuple = benchmark.label_info
        self.concept_set = sorted(set([concept for label in self.label_set for concept in label]))
        self.num_concepts = len(self.concept_set)
        self.concept_map = {concept: idx for idx, concept in enumerate(self.concept_set)}

        self.original_classes_in_exp = self.benchmark.original_classes_in_exp
        self.classes_in_exp = self.benchmark.classes_in_exp
        self.map_related_class_to_origin = dict()   # {task_id: {related_cls: origin_cls}}
        for exp_id in range(len(self.classes_in_exp)):
            task_id = exp_id if self.benchmark.return_task_id else 0
            if task_id not in self.map_related_class_to_origin.keys():
                self.map_related_class_to_origin[task_id] = dict()
            for cls_id, cls in enumerate(self.classes_in_exp[exp_id]):
                self.map_related_class_to_origin[task_id][cls.item()] = self.original_classes_in_exp[
                    exp_id, cls_id].item()
        # print(f'debug: map_related_class_to_origin: \n{self.map_related_class_to_origin}')

    def map_label_to_origin(self, batched_label, batched_task_id):
        """Map related label back to its original value.
        :param batched_label: cuda Long Tensor [bs,]
        :param batched_task_id: cuda Long Tensor [bs,]
        """
        # original_labels = [
        #     original_classes_in_exp[
        #         task_id, np.where(classes_in_exp[task_id] == re_label)[0][0]
        #     ] for re_label, task_id in zip(batched_label.tolist(), batched_task_id.tolist())]
        original_labels = [
            self.map_related_class_to_origin[task_id][re_label]
            for re_label, task_id in zip(
                batched_label.tolist(), batched_task_id.tolist())
        ]

        # print(f'\ndebug: original_labels: \n{original_labels}')

        concept_comb_strs = [self.map_int_label_to_tuple[cls_idx] for cls_idx in original_labels]
        # [('leaves', 'shirt'), ('grass', 'table')]
        concept_comb_ints = [[
            self.concept_map[concept] for concept in self.map_int_label_to_tuple[cls_idx]
        ] for cls_idx in original_labels]
        # [[11, 15], [7, 19]]

        return concept_comb_ints, concept_comb_strs

    def map_origin_label_to_multi_concept_labels(self, concept_comb_ints):
        """To vary-hot repr
        :param concept_comb_ints: list of [bs, n_concept(can be different for different sample)]
        """

        multi_label_targets = torch.cat([
            torch.zeros(1, self.num_concepts).scatter_(
                1, torch.tensor(concept_comb).unsqueeze(0), 1.) for concept_comb in concept_comb_ints
        ])
        # [bs, num_concepts]    0 or 1 vary-hot cpu

        return multi_label_targets

    def before_backward(self, strategy, **kwargs):
        """
        Calculate multi_label_loss and add into strategy.loss
        """

        mb_x = strategy.mb_x
        mb_y = strategy.mb_y
        mb_task_id = strategy.mb_task_id

        concept_comb_ints, concept_comb_strs = self.map_label_to_origin(mb_y, mb_task_id)
        multi_label_targets = self.map_origin_label_to_multi_concept_labels(concept_comb_ints).to(mb_y.device)
        # [bs, num_concepts]

        logits = self.model.forward_multi_class(mb_x)
        loss = self.criterion(logits, multi_label_targets)

        if self.mask_origin_loss:
            strategy.loss = self.weight * loss
        else:
            strategy.loss = strategy.loss + self.weight * loss
