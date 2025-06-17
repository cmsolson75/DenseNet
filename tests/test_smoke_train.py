from hydra import initialize, compose
from hydra.utils import instantiate
from lightning import Trainer


def test_training_smoke():
    with initialize(config_path="../conf", version_base="1.3"):
        # cfg = compose(config_name="test.yaml")
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                "dataset=test_dataset",
                "seed=42",
                "debug=true",
                "ckpt_path=null",
            ]
        )

        model = instantiate(cfg.model)
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
        lightning_module = instantiate(
            cfg.training,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        datamodule = instantiate(cfg.dataset)
        logger = instantiate(cfg.logger)
        callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]

        trainer = Trainer(
            logger=logger,
            callbacks=callbacks,
            **cfg.trainer,
            fast_dev_run=True,
        )

        trainer.fit(lightning_module, datamodule=datamodule)
