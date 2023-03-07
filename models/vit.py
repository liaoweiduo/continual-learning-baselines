"""
This code was based on the file vit.py (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
from the lucidrains/vit-pytorch library (https://github.com/lucidrains/vit-pytorch).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn
from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from collections import OrderedDict

'''
Modification: change all nn.LayerNorm -> nn.BatchNorm1d or nn.BatchNorm1d? 
'''

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, bn_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm = nn.BatchNorm1d(bn_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class T_block(nn.Module):
    def __init__(self, dim, bn_dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn = PreNorm(dim, bn_dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff = PreNorm(dim, bn_dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, bn_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(T_block(dim, bn_dim, heads, dim_head, mlp_dim, dropout))
    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x

class ViT(nn.Module):
    """
    MODIFY: remove Linear in mlp_head
    """
    def __init__(self, *, image_size, patch_size,
                 # num_classes,
                 dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim = dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            # nn.BatchNorm1d(num_patches),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # nn.BatchNorm1d(num_patches),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, num_patches + 1, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.BatchNorm1d(dim),
            # nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    @property
    def output_size(self):
        return self.dim


class DViT(DynamicModule):
    """
    ViT with classifier
    """
    def __init__(self, image_size=128,
                 patch_size=16, dim=1024, depth=9, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                 initial_out_features: int = 2,
                 pretrained=False, pretrained_model_path=None, fix=False):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.classifier = IncrementalClassifier(self.vit.output_size, initial_out_features=initial_out_features)

        if pretrained:
            print('Load pretrained ViT model from {}'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.vit.load_state_dict(ckpt_dict['state_dict'])
            else:   # load vit and classifier
                self.load_state_dict(ckpt_dict)

            # Freeze the parameters of the feature extractor
            if fix:
                for param in self.vit.parameters():
                    param.requires_grad = False

    def forward(self, x):
        out = self.vit(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class MTViT(MultiTaskModule, DynamicModule):
    """
    MultiTask ViT
    It employs multi-head output layer
    """
    def __init__(self, image_size=128,
                 patch_size=16, dim=1024, depth=9, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                 initial_out_features: int = 2,
                 pretrained=False, pretrained_model_path=None, fix=False):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        self.classifier = MultiHeadClassifier(self.vit.output_size, initial_out_features=initial_out_features)

        if pretrained:
            print('Load pretrained ViT model from {}'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.vit.load_state_dict(ckpt_dict['state_dict'])
            else:   # load vit and classifier
                d = OrderedDict()
                for key, item in ckpt_dict.items():
                    if key.startswith('vit'):
                        d['.'.join(key.split('.')[1:])] = item
                self.vit.load_state_dict(d)
                # self.load_state_dict(ckpt_dict)

            # Freeze the parameters of the feature extractor
            if fix:
                for param in self.vit.parameters():
                    param.requires_grad = False

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        out = self.vit(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_label)


def get_vit(
        image_size=128,
        multi_head: bool = False,
        initial_out_features: int = 2, pretrained=False, pretrained_model_path=None, fix=False,
        patch_size=16, dim=1024, depth=9, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1
):
    if multi_head:
        return MTViT(image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                     dropout=dropout, emb_dropout=emb_dropout,
                     initial_out_features=initial_out_features,
                     pretrained=pretrained, pretrained_model_path=pretrained_model_path, fix=fix)
    else:
        return DViT(image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                    dropout=dropout, emb_dropout=emb_dropout,
                    initial_out_features=initial_out_features,
                    pretrained=pretrained, pretrained_model_path=pretrained_model_path, fix=fix)


__all__ = ['DViT', 'MTViT', 'get_vit']


if __name__ == '__main__':

    from models import get_parameter_number

    # from vit_pytorch import ViT
    v = ViT(
        image_size=128,     # org: 256
        patch_size=16,      # org: 32
        # num_classes=2,
        dim=512,   # org: 1024  512: same as resnet-18
        depth=5,        # org: 6
        heads=16,        # org: 16
        mlp_dim=512,       # org: 2048
        dropout=0.1,
        emb_dropout=0.1
    )
    # v = ViT(      # same param size (10MB) as resnet-18
    #     image_size=128,     # org: 256
    #     patch_size=16,      # org: 32
    #     # num_classes=2,
    #     dim=512,   # org: 1024  512: same as resnet-18
    #     depth=5,        # org: 6
    #     heads=8,        # org: 16
    #     mlp_dim=1024,       # org: 2048
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    img = torch.randn(1, 3, 128, 128)
    preds = v(img)  # (1, 1000)

    # self.to_patch_embedding = nn.Sequential(
    #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
    #     # nn.BatchNorm1d(num_patches),
    #     nn.LayerNorm(patch_dim),
    #     nn.Linear(patch_dim, dim),
    #     # nn.BatchNorm1d(num_patches),
    #     nn.LayerNorm(dim),
    # )
    # x1 = v.to_patch_embedding[0](img)
    # b, n, _ = x1.shape
    # print(b, n)
    # x2 = v.to_patch_embedding[1](x1)
    # b, n, _ = x2.shape
    # print(b, n)

    # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    # x = torch.cat((cls_tokens, x), dim=1)
    # x += self.pos_embedding[:, :(n + 1)]
    # x = self.dropout(x)
    #
    # x = self.transformer(x)
    #
    # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
    #
    # x = self.to_latent(x)
    # return self.mlp_head(x)

    d = get_parameter_number(v)
    print(d)

    print(f'Total number of parameters: {d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Trainable"] * 4 / 1024 / 1024:.2f}MB')


    # 10.43MB
