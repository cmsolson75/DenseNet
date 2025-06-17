import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = paths["root"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.normalize = normalize
        self.input_shape = input_shape

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_shape[1], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
        self.val_dataset = CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
    

class CIFAR100DataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = paths["root"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(input_shape[1], padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
        self.val_dataset = CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class SVHNDataModule(L.LightningDataModule):
    def __init__(
        self,
        paths: dict,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        normalize: dict,
        input_shape: list,
    ) -> None:
        super().__init__()
        self.data_dir = paths["root"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.transform_train = transforms.Compose(
                    [
                        transforms.RandomCrop(input_shape[1], padding=2),  # reduced padding
                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
                    ]
                )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
    def prepare_data(self):
        SVHN(self.data_dir, train=True, download=True)
        SVHN(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = SVHN(self.data_dir, train=True, transform=self.transform_train)
        self.val_dataset = SVHN(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

