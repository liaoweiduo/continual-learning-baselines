"""
Supervised Contrastive Learning, NeurIPS 2020.

Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]      # num_view
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)      # flat with order 1st, 2nd view...
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]     # only the 1st view as anchors
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   # all img with the order all 1st, all 2nd, all 3rd view ...
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()       # / 10      # for scaling to 0-1

        return loss


if __name__ == "__main__":
    torch.random.manual_seed(0)
    criterion = SupConLoss()
    _features = torch.randn(6, 1, 8)        # bs 6, n_view 1?, dim 8
    _features = _features / torch.norm(_features, dim=2, p=2, keepdim=True)     # norm with Euclidean dis = 1
    _labels = torch.Tensor([0, 0, 1, 1, 2, 2]).long()
    _loss1 = criterion(_features, _labels)
    # _loss1 = tensor(9.2037)

    torch.random.manual_seed(0)
    criterion = SupConLoss()
    _features = torch.randn(3, 2, 8)        # bs 3, n_view 2, dim 8
    _features = _features / torch.norm(_features, dim=2, p=2, keepdim=True)     # norm with Euclidean dis = 1
    _labels = torch.Tensor([0, 1, 2]).long()
    _loss2 = criterion(_features, _labels)
    # _loss2 = tensor(9.2037)

    torch.random.manual_seed(0)
    criterion = SupConLoss()
    _features = torch.Tensor([
        [[0, 0, 0, 1, 1, 1, 0, 0, 0]],
        [[0, 0, 0, 1, 1, 1, 0, 0, 0]],
        [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
        [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
    ]).cuda().requires_grad_(True)
    _features = _features / torch.norm(_features, dim=2, p=2, keepdim=True)  # norm with Euclidean dis = 1
    _labels = torch.Tensor([0, 0, 1, 1, 2, 2]).long().cuda()
    _loss3 = criterion(_features, _labels)
    # _loss3 = tensor(2.4239e-06)

    torch.random.manual_seed(0)
    criterion = SupConLoss()
    _features = torch.Tensor([
        [[0, 0, 0, 1, 1, 1, 0, 0, 0]],
        [[0, 1, 0, 1, 1, 1, 0, 0, 0]],
        [[1, 1, 1, 0, 0, 0, 0, 0, 0]],
        [[1, 1, 1, 0, 1, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 1, 1]],
        [[0, 1, 0, 0, 0, 0, 1, 1, 1]],
    ]).cuda().requires_grad_(True)
    _features = _features / torch.norm(_features, dim=2, p=2, keepdim=True)  # norm with Euclidean dis = 1
    _labels = torch.Tensor([0, 0, 1, 1, 2, 2]).long().cuda()
    _loss4 = criterion(_features, _labels)
    # _loss4 = tensor(0.0022)

    torch.random.manual_seed(0)
    criterion = SupConLoss()
    _features = torch.Tensor([
        [[0, 0, 1, 1, 0, 1, 0, 0, 1]],
        [[0, 1, 0, 1, 1, 1, 0, 0, 0]],
        [[1, 1, 0, 0, 1, 0, 0, 0, 0]],
        [[1, 1, 1, 0, 1, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 1, 1, 1]],
        [[0, 1, 0, 0, 0, 0, 1, 1, 1]],
        [[1, 0, 0, 1, 0, 1, 0, 1, 1]],
    ]).cuda().requires_grad_(True)
    _features = _features / torch.norm(_features, dim=2, p=2, keepdim=True)  # norm with Euclidean dis = 1
    _labels = torch.Tensor([0, 0, 1, 1, 2, 2, 2]).long().cuda()
    _loss5 = criterion(_features, _labels)
    # _loss5 = tensor(1.7191, device='cuda:0', grad_fn=<MeanBackward0>)
