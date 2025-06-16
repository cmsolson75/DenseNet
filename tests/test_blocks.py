import torch
from src.models.blocks import _DenseLayer, DenseBlock, Transition

def test_dense_layer():
    layer = _DenseLayer(64, 32, 4, 0.0)
    x = torch.randn(2, 64, 32, 32)
    y = layer(x)
    assert y.shape == (2, 32, 32, 32)

def test_dense_block():
    block = DenseBlock(64, 32, 1, 4, 0.0)
    x = torch.randn(2, 64, 32, 32)
    y = block(x)
    assert y.shape == (2, 96, 32, 32)

def test_transition():
    layer = Transition(128, compression_factor=0.5)
    x = torch.randn(2, 128, 32, 32)
    y = layer(x)
    assert y.shape == (2, 64, 16, 16)
