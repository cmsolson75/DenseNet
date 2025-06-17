import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
    
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    datamodule = instantiate(cfg.dataset)

    lightning_module = instantiate(
        cfg.training,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    logger = instantiate(cfg.logger)
    # Trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer,
    )
    if cfg.ckpt_path:
        print(f"Resuming from checkpoint: {cfg.ckpt_path}")
        trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    else:
        print("Starting training from scratch")
        trainer.fit(lightning_module, datamodule=datamodule)




if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()