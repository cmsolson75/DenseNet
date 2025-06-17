import torch
from torch import nn
from typing import Tuple
from src.models.blocks import DenseBlock, Transition


class DenseNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        block_config: Tuple[int, int, int] = (32, 32, 32),
        num_classes: int = 10,
        bottleneck_size: int = 4,
        compression_factor: float = 0.5,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes # Save for downstream usage
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                growth_rate * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(growth_rate * 2),
            nn.ReLU(inplace=True),
        )
        num_features = growth_rate * 2
        blocks = []
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_input_features=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers,
                bottleneck_size=bottleneck_size,
                drop_prob=drop_prob,
            )
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                transition_layer = Transition(
                    num_features=num_features, compression_factor=compression_factor
                )
                blocks.append(transition_layer)
                num_features = int(num_features * compression_factor)
        self.feature_extractor = nn.Sequential(*blocks)

        self.final_norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        # From official implementation
        # https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
        # https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
