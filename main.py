import torch
from src.models.densenet import DenseNet
<<<<<<< HEAD
import torchvision
from torchvision.transforms import transforms
import torch
from torch import nn


def create_cifar10_dataloader(
    train: bool, download: bool = True
) -> torch.utils.data.DataLoader:
    batch_size = 128

    root_dir = "cifar"
    num_workers = 7

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = True
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        shuffle = False

    dataset = torchvision.datasets.CIFAR10(
        root=root_dir, train=train, download=download, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


cifar10_dl = create_cifar10_dataloader(train=True, download=True)
model = DenseNet(3, 16).to("cuda")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
EPOCHS = 5

n = len(cifar10_dl)

for epoch in range(EPOCHS):
    loss_local = 0
    for x, y in cifar10_dl:
        x, y = x.to("cuda"), y.to("cuda")
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_local += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss_local / n:.4f}")
=======


x = torch.randn(1, 3, 32, 32).to("cuda")
model = DenseNet(3, 16).to("cuda")

out = model(x)
print(out)
>>>>>>> faf4a1b (Added general infrastructure and some tests)
