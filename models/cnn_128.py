from collections import OrderedDict

import torch.nn as nn
import torch
from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule

"""
For image size 128*128
"""


class CNN128(DynamicModule):
    def __init__(self, num_classes: int = None, pretrained=False, pretrained_model_path=None):
        super().__init__()
        layers = nn.Sequential(*(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4*4*1024, 1024),
            nn.ReLU(inplace=True),
        ))
        self.features = nn.Sequential(*layers)
        if num_classes is None:
            self.classifier = IncrementalClassifier(1024)
        else:
            self.classifier = nn.Linear(1024, num_classes)

        if pretrained:
            print('Load pretrained cnn model from {}.'.format(pretrained_model_path))
            self._load_pretrained_feature_extractor(pretrained_model_path)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _load_pretrained_feature_extractor(self, pretrained_model_path):
            ckpt_dict = torch.load(pretrained_model_path)
            if 'state_dict' in ckpt_dict:
                self.features.load_state_dict(ckpt_dict['state_dict'])
            else:
                d = OrderedDict()
                for key, item in ckpt_dict.items():
                    if key.startswith('features'):
                        d['.'.join(key.split('.')[1:])] = item
                self.features.load_state_dict(d)

            # Freeze the parameters of the feature extractor
            for param in self.features.parameters():
                param.requires_grad = False


class MTCNN128(MultiTaskModule, DynamicModule):
    def __init__(self, pretrained=False, pretrained_model_path=None):
        super().__init__()
        layers = nn.Sequential(*(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4*4*1024, 1024),
            nn.ReLU(inplace=True),
        ))
        self.features = nn.Sequential(*layers)
        self.classifier = MultiHeadClassifier(1024, initial_out_features=5)

        if pretrained:
            print('Load pretrained cnn model from {}.'.format(pretrained_model_path))
            self._load_pretrained_feature_extractor(pretrained_model_path)

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x, task_label)
        return x

    def _load_pretrained_feature_extractor(self, pretrained_model_path):
            ckpt_dict = torch.load(pretrained_model_path)
            if 'state_dict' in ckpt_dict:
                self.features.load_state_dict(ckpt_dict['state_dict'])
            else:
                d = OrderedDict()
                for key, item in ckpt_dict.items():
                    if key.startswith('features'):
                        d['.'.join(key.split('.')[1:])] = item
                self.features.load_state_dict(d)

            # Freeze the parameters of the feature extractor
            for param in self.features.parameters():
                param.requires_grad = False


__all__ = ['CNN128', 'MTCNN128']

if __name__ == '__main__':
    model = CNN128()
    print(model.features(torch.zeros((1, 3, 128, 128))).shape)      # 1024
