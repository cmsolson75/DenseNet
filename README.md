# DenseNet-BC implementation
Project quality
- Add simple PyTest setup with ini
    - Shape tests
    - Forward pass sanity
    - Loss Reduction
    - Config instantiation
    - Training loop smoke test
- Add pre commit
- Add Hydra
- Add Optuna
- Add W&B
- Add PTL


References
- https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py


Paper implementation
* DenseNet-BC (k = 12)
    * CIFAR10
    * I had more params then they did with there implementation, must have mixed up layers, I did 3 dense blocks each with 32 dense layers. That creates 96 layers, and if you add the stem and classification head then you get 100 layuers, maybe Pytorch is different.
    * They have 0.8M params, I have 3.5M
    * My error rate was 4.54 vs the papers 4.51, this is in the margen of error
    * I MADE A MISTAKE: I need to do this again with a dense size of 16, if its 32 then we get a densenet 190



EXPERIMENTS
* DenseNet-BC (k=12): 100 layers (16 per block chunk)
    * C10+
    * C100+
    * SVHN
* Use CosineAnnealingLR
* 300 Epochs