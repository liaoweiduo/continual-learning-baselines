################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-3-2023                                                              #
# Author(s): Weiduo Liao                                                       #
# E-mail: liaowd@mail.sustech.edu.cn                                           #
################################################################################
""" README
This code defines the components for a modularity network.
"""
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule


class BackboneModule(nn.Module):
    """ manage modules in one layer for backbones.
    """
    def __init__(self, in_channels, out_channels, module_arch='resnet18_pnf', layer_idx=0, dropout=0,
                 multi_head=1, identity=True):
        super(BackboneModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.module_arch = module_arch
        self.layer_idx = layer_idx
        self.multi_head = multi_head        # num of modules in this layer
        self.identity = identity            # true if the last module is identity module
        if 'resnet18' == module_arch:
            from resnet import conv1x1, BasicBlock

            def _make_layer(inplanes, planes, blocks, stride=1, out_planes=None):
                if out_planes is None:
                    out_planes = planes
                block = BasicBlock
                downsample = None
                if stride != 1 or inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(inplanes, planes * block.expansion, stride),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                if blocks == 1:     # only one block, use out_planes
                    layers.append(block(inplanes, out_planes, stride, downsample))
                else:
                    layers.append(block(inplanes, planes, stride, downsample))
                inplanes = planes * block.expansion
                for i in range(1, blocks):
                    if i < blocks - 1:
                        layers.append(block(inplanes, planes))
                    else:
                        layers.append(block(inplanes, out_planes))
                        # todo: only the last conv has the out_planes,
                        #  currently two conv all has the out_planes

                return nn.Sequential(*layers)

            num_blocks = self.multi_head - 1 if self.identity else self.multi_head
            self._blocks = nn.ModuleList(
                _make_layer(
                    self.in_channels, planes=self.out_channels,
                    blocks=2, stride=1 if layer_idx == 0 else 2,
                    out_planes=self.out_channels // self.multi_head)
                for _ in range(num_blocks)
            )
            self._identity = _make_layer(
                    self.in_channels, planes=self.out_channels,
                    blocks=2, stride=1 if layer_idx == 0 else 2,
                    out_planes=self.out_channels // self.multi_head) if self.identity else None
            self.freeze(-1)     # freeze the identity module

        elif 'resnet18_pnf' == module_arch:
            from resnet18_pnf import conv1x1, BasicBlockFilm

            def _make_layer(inplanes, planes, blocks, stride=1, _multi_head=1):
                block = BasicBlockFilm
                downsample = None
                if stride != 1 or inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(inplanes, planes * block.expansion, stride),
                        nn.BatchNorm2d(planes * block.expansion))

                layers = []
                layers.append(block(inplanes, planes, stride=stride, downsample=downsample, multi_head=_multi_head))
                inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(inplanes, planes, multi_head=_multi_head))

                # return nn.Sequential(*layers)
                # the input also takes head_idx, which is not capable for nn.Sequential
                return layers

            num_blocks = self.multi_head - 1 if self.identity else self.multi_head
            self._blocks = nn.ModuleList(_make_layer(
                self.in_channels, planes=self.out_channels,
                blocks=2, stride=1 if layer_idx == 0 else 2, _multi_head=num_blocks))
            self._identity = nn.Sequential(*_make_layer(
                    self.in_channels, planes=self.out_channels,
                    blocks=2, stride=1 if layer_idx == 0 else 2, _multi_head=1)) if self.identity else None
            self.freeze(-1)     # freeze the identity module

        else:
            raise Exception(f'Un-implemented module_arch: {module_arch}.')

        self.use_dropout = False
        if dropout > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, module_idx=None):
        if module_idx is None:
            # return all outs
            out = []
            for module_idx in range(self.multi_head):
                module_out = self.forward_single(x, module_idx)
                out.append(module_out)
            return out
        else:
            return self.forward_single(x, module_idx)

    def forward_single(self, x, module_idx):
        assert (module_idx < self.multi_head
                ), f'module_idx: {module_idx} exceeds num of module in this layer: {self.layer_idx}.'

        if self.identity and module_idx == self.multi_head - 1:     # forward identity module.
            x = self._identity(x)
        elif self.module_arch == 'resnet18_pnf':
            for block_idx in range(len(self._blocks)):
                x = self._blocks[block_idx](x, module_idx)
        else:
            x = self._blocks[module_idx](x)

        if self.use_dropout:
            x = self.dropout(x)

        return x

    def freeze(self, module_idx: Optional[int] = None):
        if module_idx is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            module_idx = self.multi_head - 1 if module_idx == -1 else module_idx
            assert (self.module_arch != 'resnet18_pnf' or (self.identity and module_idx == self.multi_head - 1)
                    ), f"resnet18_pnf can not freeze not identity module."
            if self.identity and module_idx == self.multi_head - 1:
                target = self._identity
            else:
                assert (module_idx < len(self._blocks)
                        ), f'module_idx: {module_idx} exceeds num of module in this layer: {self.layer_idx}.'
                target = self._blocks[module_idx]
            for param in target.parameters():
                param.requires_grad = False


class SelectorModule(nn.Module):
    """ Generate one selector.
    """
    def __init__(self, in_channels, rep_dim, init_n_class,
                 mode='prototype', layer_idx=0,
                 pooling_before=False, proj=True, metric='dot', bn=True, layer_norm=True):
        super(SelectorModule, self).__init__()
        self.in_channels = in_channels
        self.rep_dim = rep_dim
        self.n_class = init_n_class     # num of concepts
        self.mode = mode
        self.layer_idx = layer_idx
        self.pooling_before = pooling_before
        self.proj = proj
        self.metric = metric
        self.bn = bn
        self.layer_norm = layer_norm  # True use layer_norm for normalizing similarity matrix, else use standard.

        if self.proj:
            self.conv1 = nn.Conv2d(in_channels, self.rep_dim, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
            if self.bn:
                self.bn1 = nn.BatchNorm2d(self.rep_dim)

        self.norm = None
        if self.layer_norm:
            self.norm = nn.LayerNorm(self.n_class)    # num of proto

        if self.mode == 'prototype':
            prot_shape = (1, 1)
            self.prototype_shape = (self.n_class, self.rep_dim, *prot_shape)
            self.prototypes = nn.Parameter(torch.rand(self.prototype_shape))
            self.logit_scale = nn.Parameter(torch.ones([]))
        else:
            raise Exception(f'Un-implemented mode: {mode}.')

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

    def forward(self, x, inference=False):
        if self.pooling_before:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        if self.proj:
            x = self._project(x)

        dist = self._distance(x)        # [bs, n_proto, H, W]

        sm = F.softmax(dist.view(*dist.size()[:2], -1), dim=2).view_as(dist)
        # [bs, n_proto, H, W]   sim over H*W   sum on dim=(3, 4) = 1

        vecs = []
        for i in range(sm.size(1)):
            smi = sm[:, i].unsqueeze(1)
            vecs.append(torch.mul(x, smi))  # batch of imgs applying sim mask
        vecs = torch.stack(vecs, dim=1).sum(dim=(3, 4))     # [bs, n_proto, rep_dim]    sum over H*W (dim 3 and 4)

        # if callable(self.metric) or self.metric == 'dot':
        dist = (vecs * self.prototypes.flatten(1)).sum(2)   # [bs, n_proto]   dot distance (* and sum) from prototypes
        # print(dist.shape)

        # if self.metric == 'euclidean':
        #     # min pooling for euclidean distance
        #     dist, idxs = F.max_pool2d(-dist, dist.shape[2:], return_indices=True)
        #     dist = -dist
        # else:
        #     dist, idxs = F.max_pool2d(dist, dist.shape[2:], return_indices=True)
        # dist = dist.flatten(1)

        out = dist
        if self.metric == 'euclidean':
            out = -out

        '''gumbel sigmoid or sigmoid'''
        # print(f'out: {out}')
        # out = out / 10    # increase sensitivity before sigmoid.
        # out is metric value, which > 0. the smallest prob is 0.5 to be select. so norm is needed.
        if self.norm is None:
            # Option: mean var norm
            out = (out - out.mean(1, keepdim=True).detach()
                   ) / torch.sqrt(out.std(1, keepdim=True).detach() ** 2 + 1e-15)
            # # Option: norm to [-1, 1].
            # out = out - out.min(1, keepdim=True)[0].detach()
            # out = out / out.max(1, keepdim=True)[0].detach()
            # out = out * 2 - 1       # always has one 1 and -1 ?
            # import torchvision.transforms as T
            # transform = T.Normalize(mean=out.mean(1).detach(), std=out.std(1).detach())
            # out = transform(out)
        else:
            out = self.norm(out)
        # print(f'out after norm: {out}')

        if inference:
            out = torch.sigmoid(out)
            out[out >= 0.5] = 1
            out[out < 0.5] = 0
            # todo: test whether use all modules is better than use threshold.
            #   since SPSnet does.
        else:
            out = self._gumbel_hard_sigmoid(out)        # [bs, n_proto]

        '''handle none selection case'''
        out_identity = torch.zeros(out.shape[0], 1).to(out.device)
        out_identity[out.detach().sum(1) == 0] = 1

        out = torch.cat([out, out_identity], dim=1)     # [bs, n_proto + 1]

        return out, sm, vecs, dist

    def _project(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        if self.bn:
            x = self.bn1(x)

        return x

    def _distance(self, x):
        if callable(self.metric):
            dist = self.metric(x)
        elif self.metric == 'dot':
            # dist = self.logit_scale.exp() * F.conv2d(x, weight=self.prototypes)
            dist = F.conv2d(x, weight=self.prototypes)
        elif self.metric == 'euclidean':
            dist = self._l2_convolution(x)
        elif self.metric == 'cosine':
            x = x / x.norm(dim=1, keepdim=True)
            weight = self.prototypes / self.prototypes.norm(dim=1, keepdim=True)
            dist = self.logit_scale.exp() * F.conv2d(x, weight=weight)
        else:
            raise NotImplementedError('Metric {} not implemented.'.format(self.metric))

        return dist

    def _l2_convolution(self, x):
        """
        Taken from https://github.com/cfchen-duke/ProtoPNet
        apply self.prototype_vectors as l2-convolution filters on input x
        """
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototypes)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _gumbel_hard_sigmoid(self, dist):
        """
        Gumber sigmoid is 2-way (value and 1-value) gumber softmax for each element.
        """
        # dist shape: [bs, n_proto]
        background_dist = dist.detach()
        background_dist = torch.sigmoid(background_dist)
        background_dist = 1 - background_dist
        background_dist = torch.logit(background_dist)      # sigmoid^(-1)
        dist = torch.stack([dist, background_dist])     # [bs, n_proto, 2]
        sm = F.gumbel_softmax(dist, dim=1, hard=True)[0]     # [bs, n_proto]

        return sm

    def add_prototypes(self):
        assert (self.layer_norm is False
                ), "Should not use layer norm for normalizing similarity matrix, if dynamically add prototypes."
        pass


if __name__ == '__main__':
    model = SelectorModule(in_channels=64, rep_dim=64, init_n_class=3, layer_norm=True)

    X = torch.randn((5, 64, 63, 63))    # after first conv2 for 3*128*128

    _out = model(X, True)

    from models import get_parameter_number

    d = get_parameter_number(model)
    # model = torch.nn.DataParallel(model)
    print(d)
    print(f'Total number of parameters: {d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Trainable"] * 4 / 1024 / 1024:.2f}MB')

