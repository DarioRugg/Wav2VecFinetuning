from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.lightning_dataloaders import DataModule

from scripts.utils import get_model, get_model_from_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

# import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

import logging
logger = logging.getLogger("__name__")

@hydra.main(config_path=Path(".", "Assets", "Config"), config_name="config.yaml")
def main(cfg: DictConfig):

    logger.info(OmegaConf.to_yaml(cfg))

    seed_everything(0)

    wandb.init(project="emoaudio")
    wandb_logger = WandbLogger(project=cfg.simulation_name)

    # ------------------> Dataset <-----------------------
    data_module = DataModule(config=cfg)

    # ------------------> Model <-----------------------
    model = get_model(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./models',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_weights_only=False
    )
    trainer = Trainer(
        logger=wandb_logger,  # W&B integration
        max_epochs=cfg.model.epochs,  # number of epochs
        callbacks=[checkpoint_callback],
        gpus=[cfg.machine.gpu] if cfg.machine.gpu is not False else None
    )

    # ------------------> Training <-----------------------
    if cfg.train:
        trainer.fit(model=model, datamodule=data_module)

    if cfg.test:
        # ------------------> Loading best model <-----------------------
        model = get_model_from_checkpoint(cfg, checkpoint_path=checkpoint_callback.best_model_path)

        # ------------------> Testing <-----------------------
        trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
