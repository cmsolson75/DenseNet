import lightning as L
import torch.nn as nn
from torchmetrics import Accuracy

class ClassifierTrainingWrapper(L.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_acc = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.model.num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc.update(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc.update(logits, y)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
