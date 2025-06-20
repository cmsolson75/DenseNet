# DenseNet-BC for CIFAR10/100 and SVHN

This repository implements a reproducible version of DenseNet-BC using PyTorch Lightning. Configuration is managed via Hydra, with integrated support for Optuna sweeps and Weights & Biases logging.

## ✅ Features

- DenseNet-BC implementation (k=12)
- Dataset support: CIFAR-10, CIFAR-100, SVHN
- Config-driven training with Hydra
- Logging via Weights & Biases
- Training via PyTorch Lightning
- CosineAnnealingLR scheduler
- Automated model checkpointing (top-k + last)
- Testing via PyTest:
  - Shape tests
  - Forward-pass sanity checks
  - Loss reduction validation
  - Config instantiation
  - Smoke test for training loop
- Pre-commit configuration

## 🧪 Experiments

| Dataset   | Configuration                | Val Error (%) | Notes                                              |
|-----------|------------------------------|---------------|----------------------------------------------------|
| CIFAR10   | DenseNet-BC (k=12, 100L)      | 5.18          | Improved over paper (4.51%) using Cosine LR       |
| CIFAR10   | DenseNet-BC (k=12, 190L)      | 4.54          | Overparameterized (3.5M vs 0.8M), matches paper    |
| CIFAR100  | DenseNet-BC (k=12, 100L)      | 27.76         | Worse than paper by ~3.3%; batch size = 256       |
| SVHN      | DenseNet-BC (k=12, 100L)      | 3.592         | Higher than paper; possibly due to batch size     |

> ⚠️ Note: 190-layer run used 32 layers per block, not 16. The correct config for 100 layers is `[16, 16, 16]`.

## 🧰 Usage

Train a model with:

```bash
python train.py experiment=train_cifar10
python train.py experiment=train_cifar100
python train.py experiment=train_svhn
```
Validate configs with:
```
pytest
```
🧪 Testing

PyTest covers:

- Output shapes
- Forward path validity
- Loss decrease behavior
- Config + training object instantiation
- Smoke training run (via `test_dataset`)

⚙️ Setup

```bash
pip install -r requirements.txt
pre-commit install
```

## 🔗 References

- [torchvision DenseNet (official)](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)
- Huang, Gao, et al. *Densely Connected Convolutional Networks*. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- PyTorch Lightning Documentation: https://lightning.ai/docs/pytorch/stable/
- Hydra Configuration Framework: https://hydra.cc/docs/intro/
- Optuna: https://optuna.org/
- Weights & Biases: https://wandb.ai/site
- Cosine Annealing LR: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html