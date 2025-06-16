import torch
from src.models.densenet import DenseNet


x = torch.randn(1, 3, 32, 32).to("cuda")
model = DenseNet(3, 16).to("cuda")

out = model(x)
print(out)
