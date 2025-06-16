import torch
import torch.nn as nn


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bottleneck_size: int,
        drop_prob: float,
    ):
        super().__init__()
        if not (0.0 <= drop_prob <= 1.0):
            raise ValueError("Dropout probability has to be between 0 and 1")
        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(
            in_channels=num_input_features,
            out_channels=bottleneck_size * growth_rate,
            kernel_size=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(bottleneck_size * growth_rate)
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_size * growth_rate,
            out_channels=growth_rate,
            kernel_size=3,
            bias=False,
            padding=1,
        )

    def forward(self, x):
        x = self.conv1(self.act(self.bn1(x)))
        x = self.dropout(self.conv2(self.act(self.bn2(x))))
        return x


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        num_layers: int,
        bottleneck_size: int,
        drop_prob: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                drop_prob=drop_prob,
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        new_features = None
        for layer in self.layers:
            all_features = torch.cat(features, 1)
            new_features = layer(all_features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Module):
    def __init__(
        self,
        num_features: int,
        compression_factor: float = 0.5,
    ) -> None:
        super().__init__()
        if not (0.0 <= compression_factor <= 1.0):
            raise ValueError("Compression Factor must be between 0 and 1")
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                out_channels=int(num_features * compression_factor),
                kernel_size=1,
                bias=False,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition(x)
