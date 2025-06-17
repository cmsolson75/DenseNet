from hydra import initialize, compose
from hydra.utils import instantiate

def test_cfg_model_instantiation():
    with initialize(config_path="../conf", version_base="1.3"):
        cfg = compose(config_name="config.yaml")
        model = instantiate(cfg.model)
        assert callable(model)


def test_cfg_training_instantiation():
    with initialize(config_path="../conf", version_base="1.3"):
        cfg = compose(config_name="config.yaml")
        
        model = instantiate(cfg.model)
        assert hasattr(model, "parameters"), "Model is not a torch.nn.Module"
        assert len(list(model.parameters())) > 0, "Model has no parameters"

        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
        
        lightning_module = instantiate(
            cfg.training,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
        assert hasattr(lightning_module, "training_step")