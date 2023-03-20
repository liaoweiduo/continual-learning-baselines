################################################################################
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 15-3-2023                                                              #
# Author(s): Weiduo Liao                                                       #
# E-mail: liaowd@mail.sustech.edu.cn                                           #
################################################################################
""" README
This code defines a modularity network.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule

# from models.module_net_component import BackboneModule, SelectorModule
from module_net_component import BackboneModule, SelectorModule
from resnet18_pnf import CatFilm


class ModuleNet(nn.Module):

    def __init__(
            self, num_layers=4, backbone_arch='resnet18_pnf', selector_mode='prototype',
            image_size=128, in_channels=3, init_modules=7,
    ):
        super(ModuleNet, self).__init__()
        self.num_layers = num_layers
        self.backbone_arch = backbone_arch
        self.selector_mode = selector_mode
        self.image_size = image_size
        self.in_channels = in_channels  # img channels
        self.init_modules = init_modules   # num of modules init (init_modules + 1 modules in each layer)
        self.inplanes = 64      # dim after encoder

        assert (self.num_layers == 4
                ), f"Number of layers should be 4, {self.num_layers} instead."
        self.layer_out_dims = [64, 128, 256, 512]

        '''encoder & after backbone'''
        if 'resnet18' == self.backbone_arch:
            self.encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False),    # conv1
                nn.BatchNorm2d(self.inplanes),  # bn1
                nn.ReLU(inplace=True),  # relu
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # maxpool
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif 'resnet18_pnf' == self.backbone_arch:
            self.encoder = nn.Sequential(
                # CatFilm(self.in_channels),     # film norm
                nn.Conv2d(self.in_channels, self.inplanes, kernel_size=5, stride=2, padding=1, bias=False),    # conv1
                nn.BatchNorm2d(self.inplanes),  # bn1
                CatFilm(self.inplanes),     # film norm
                nn.ReLU(inplace=True),  # relu
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # maxpool
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.encoder = None
            self.avgpool = nn.Identity()

        '''backbone'''
        self.backbone_constructor = BackboneModule
        self.backbone = nn.ModuleList()
        self.init_backbone()

        '''selector'''
        self.selector = nn.ModuleList()
        self.selector_constructor = SelectorModule
        self.init_selector()

    def init_backbone(self):
        self.backbone = nn.ModuleList()
        in_channels = self.inplanes
        for layer_idx in range(self.num_layers):
            self.backbone.append(
                self.backbone_constructor(
                    in_channels=in_channels,
                    out_channels=self.layer_out_dims[layer_idx],
                    module_arch=self.backbone_arch,
                    layer_idx=layer_idx,
                    multi_head=self.init_modules + 1,      # 8
                )
            )
            if self.backbone_arch == 'resnet18':
                in_channels = (self.layer_out_dims[layer_idx] // (self.init_modules + 1)) * (self.init_modules + 1)
            else:       # elif 'resnet18_pnf' == self.backbone_arch:
                in_channels = self.layer_out_dims[layer_idx]

    def init_selector(self):
        self.selector = nn.ModuleList()
        in_channels = self.inplanes
        for layer_idx in range(self.num_layers):
            rep_dim = self.layer_out_dims[layer_idx] // (self.init_modules + 1) \
                if self.backbone_arch == 'resnet18' else self.layer_out_dims[layer_idx]      # 64 // 8
            self.selector.append(
                self.selector_constructor(
                    in_channels=in_channels,
                    rep_dim=rep_dim,
                    init_n_class=self.init_modules,
                    mode=self.selector_mode,
                    layer_idx=layer_idx
                )
            )
            if self.backbone_arch == 'resnet18':
                in_channels = (self.layer_out_dims[layer_idx] // (self.init_modules + 1)) * (self.init_modules + 1)
            else:       # elif 'resnet18_pnf' == self.backbone_arch:
                in_channels = self.layer_out_dims[layer_idx]

    def add_modules(self, layer_idx):
        """ Add module (and init a prototype if necessary) at specific layer_idx
        """
        raise Exception("Un-implemented func: add_modules")

    def forward(self, x, inference=False):
        """
        """
        x = self.encoder(x)

        for layer_idx in range(self.num_layers):
            # selector
            selected_idxs, sm, _, _ = self.selector[layer_idx](x, inference)
            # selected_idxs [bs, n_modules + 1]

            # module
            out = self.backbone[layer_idx](x)
            for module_idx in range(self.init_modules + 1):
                masked_out = out[module_idx] * selected_idxs[:, module_idx].view(selected_idxs.shape[0], 1, 1, 1)
                out[module_idx] = masked_out

            if 'resnet18' == self.backbone_arch:
                x = torch.cat(out, dim=1)     # [bs, dim_out, H, W]
            elif 'resnet18_pnf' == self.backbone_arch:
                x = torch.sum(torch.stack(out), dim=0)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == '__main__':
    model = ModuleNet(init_modules=7)

    X = torch.randn((3, 3, 128, 128))

    Out = model(X)

    from models import get_parameter_number

    d = get_parameter_number(model)
    # model = torch.nn.DataParallel(model)
    print(d)
    print(f'Total number of parameters: {d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Trainable"] * 4 / 1024 / 1024:.2f}MB')

