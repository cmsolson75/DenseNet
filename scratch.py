import torch
from torch import nn

class DenseNode(nn.Module):
    # Something that implements the BottleNeck block
    # The Dense Block is a collection of DenseNodes
    def __init__(self, in_planes: int, planes: int):
        super().__init__()
        self.conv1 = nn.Conv2d()

    def forward(self, x):
        return x


# Create DenseBlock
class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def forward(self, x):
        state = [x]
        out = None
        for l in self.layers:
            inp = torch.cat(state, 1)
            out = l(inp)
            state.append(out)
        return out

class Transition(nn.Module):
    pass
