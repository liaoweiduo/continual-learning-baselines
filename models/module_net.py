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

from models.module_net_component import BackboneModule, SelectorModule
from models.resnet18_pnf import CatFilm


class ModuleNetBackbone(nn.Module):

    def __init__(
            self, args, backbone_arch='vit_ia3', selector_mode='prototype',
            in_channels=3, init_modules=7,
    ):
        super(ModuleNetBackbone, self).__init__()
        self.args = args
        self.num_layers = args['vit_depth'] if 'vit' in backbone_arch else 4    # 4 for resnet18
        self.backbone_arch = backbone_arch
        self.selector_mode = selector_mode
        self.image_size = args['image_size']
        self.in_channels = in_channels  # img channels
        self.init_modules = init_modules   # num of modules init (init_modules + 1 modules in each layer)

        if 'resnet18' in backbone_arch:
            self.inplanes = 64      # dim after encoder
            assert (self.num_layers == 4
                    ), f"Number of layers should be 4, {self.num_layers} instead."
            self.layer_out_dims = [64, 128, 256, 512]
        else:   # vit
            self.inplanes = self.args['vit_dim']      # dim after encoder
            self.layer_out_dims = [self.args['vit_dim']] * self.num_layers  # [384, 384,..., 384]  shape [9]

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
        elif 'vit_ia3' == self.backbone_arch:
            from models.vit import Encoder, pair

            self.dim = args['vit_dim']
            self.heads = args['vit_heads']
            self.mlp_dim = args['vit_mlp_dim']
            self.pool = 'cls'
            image_height, image_width = pair(self.image_size)
            patch_height, patch_width = pair(args['vit_patch_size'])
            self.dropout = args['vit_dropout']
            self.emb_dropout = args['vit_emb_dropout']

            assert (image_height % patch_height == 0 and image_width % patch_width == 0
                    ), 'Image dimensions must be divisible by the patch size.'

            self.num_patches = (image_height // patch_height) * (image_width // patch_width)
            self.patch_dim = self.in_channels * patch_height * patch_width
            assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

            self.encoder = Encoder(patch_height, patch_width,
                                   self.patch_dim, self.dim, self.num_patches, self.emb_dropout)

            self.avgpool = nn.LayerNorm(self.dim)
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

        '''intermediate variables for reg'''
        self.selected_idxs_each_layer = []      # reset every iteration.
        self.similarity_tensor = []             # n_layer*[bs, n_proto, Hl, Wl]

    def init_backbone(self):
        self.backbone = nn.ModuleList()
        in_channels = self.inplanes
        for layer_idx in range(self.num_layers):
            params = {
                    'in_channels': in_channels,
                    'out_channels': self.layer_out_dims[layer_idx],
                    'module_arch': self.backbone_arch,
                    'layer_idx': layer_idx,
                    'multi_head': self.init_modules + 1,      # 8
            }
            if 'vit' in self.backbone_arch:
                params.update({
                    'num_patches': self.num_patches,
                    'heads': self.heads,
                    'dim_head': 64,
                    'mlp_dim': self.mlp_dim,
                    'vit_dropout': self.dropout,
                })
            self.backbone.append(
                self.backbone_constructor(**params)
            )
            if self.backbone_arch == 'resnet18':
                in_channels = (self.layer_out_dims[layer_idx] // (self.init_modules + 1)) * (self.init_modules + 1)
            else:       # elif 'resnet18_pnf' == self.backbone_arch: or vit_ia3
                in_channels = self.layer_out_dims[layer_idx]

    def init_selector(self):
        self.selector = nn.ModuleList()
        in_channels = self.inplanes
        for layer_idx in range(self.num_layers):
            rep_dim = self.layer_out_dims[layer_idx] // (self.init_modules + 1) \
                if self.backbone_arch == 'resnet18' else self.layer_out_dims[layer_idx]      # 64 // 8
            proj = True if 'resnet18' in self.backbone_arch else False
            self.selector.append(
                self.selector_constructor(
                    in_channels=in_channels,
                    rep_dim=rep_dim,
                    init_n_class=self.init_modules,
                    mode=self.selector_mode,
                    layer_idx=layer_idx,
                    proj=proj,
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

    def forward(self, x):
        """
        """
        x = self.encoder(x)

        self.selected_idxs_each_layer = []
        self.similarity_tensor = []
        for layer_idx in range(self.num_layers):
            # selector
            selected_idxs, sm, _, _ = self.selector[layer_idx](x)
            # selected_idxs [bs, n_modules + 1]
            # sm [bs, n_proto, Hl, Wl]
            self.selected_idxs_each_layer.append(selected_idxs)
            self.similarity_tensor.append(sm)

            # module
            out = self.backbone[layer_idx](x)
            for module_idx in range(self.init_modules + 1):
                masked_out = out[module_idx] * selected_idxs[:, module_idx].view(
                    selected_idxs.shape[0], *[1]*(len(x.shape) - 1))
                out[module_idx] = masked_out

            if 'resnet18' == self.backbone_arch:
                x = torch.cat(out, dim=1)     # [bs, dim_out, H, W]
            elif self.backbone_arch in ['resnet18_pnf', 'vit_ia3']:
                x = torch.sum(torch.stack(out), dim=0)

        if self.backbone_arch == 'vit_ia3':
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    @property
    def output_size(self):
        return self.layer_out_dims[-1]


class ModuleNet(DynamicModule):
    """ ModuleNet with incremental classifier.
    """
    def __init__(self, args, initial_out_features: int = 2, pretrained=False, pretrained_model_path=None, fix=False):
        super().__init__()
        self.backbone = ModuleNetBackbone(args)
        self.classifier = IncrementalClassifier(self.backbone.output_size, initial_out_features=initial_out_features)

        if pretrained:
            print('Load pretrained ModuleNet model from {}.'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.backbone.load_state_dict(ckpt_dict['state_dict'])
            else:   # load backbone and classifier
                self.load_state_dict(ckpt_dict)

            # Freeze the parameters of the feature extractor
            if fix:
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class MTModuleNet(MultiTaskModule, DynamicModule):
    """ ModuleNet with multitask classifier.
    """

    def __init__(self, args, initial_out_features: int = 2, pretrained=False, pretrained_model_path=None,
                 fix=False, load_classifier=False):
        super().__init__()
        self.backbone = ModuleNetBackbone(args)
        self.classifier = MultiHeadClassifier(self.backbone.output_size, initial_out_features=initial_out_features)

        if pretrained:
            print('Load pretrained ModuleNet model from {}.'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.backbone.load_state_dict(ckpt_dict['state_dict'])
            else:   # load backbone and classifier
                self.load_state_dict(ckpt_dict)

            # Freeze the parameters of the feature extractor
            if fix:
                for param in self.backbone.parameters():
                    param.requires_grad = False

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_label)


def get_module_net(
        args,
        multi_head: bool = False,
        initial_out_features: int = 2, pretrained=False, pretrained_model_path=None, fix=False, load_classifier=False,
        ):
    if multi_head:
        model = MTModuleNet(args, initial_out_features, pretrained, pretrained_model_path, fix)
    else:
        model = ModuleNet(args, initial_out_features, pretrained, pretrained_model_path, fix)

    return model


__all__ = ['ModuleNetBackbone', 'ModuleNet', 'MTModuleNet', 'get_module_net']


if __name__ == '__main__':
    _args = {
        'image_size': 224,
        'model_backbone': 'vit',
        'vit_patch_size': 16,
        'vit_dim': 384,
        'vit_depth': 9,
        'vit_heads': 16,
        'vit_mlp_dim': 1536,
        'vit_dropout': 0.1,
        'vit_emb_dropout': 0.1,
    }
    model = ModuleNetBackbone(_args)

    X = torch.randn((4, 3, 224, 224))

    _out = model(X)

    from models._models import get_parameter_number

    _d = get_parameter_number(model)
    # model = torch.nn.DataParallel(model)
    print(_d)
    print(f'Total number of parameters: {_d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {_d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {_d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {_d["Trainable"] * 4 / 1024 / 1024:.2f}MB')
