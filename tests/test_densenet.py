import torch
from src.models.densenet import DenseNet

def test_densenet_forward():
    model = DenseNet(
        in_channels=3,
        growth_rate=32,
        block_config=(6, 12, 24),
        num_classes=10,
        bottleneck_size=4,
        compression_factor=0.5,
        drop_prob=0.1,
    )
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 10)