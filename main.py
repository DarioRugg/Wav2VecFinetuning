from pathlib import Path
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from scripts.lightning_dataloaders import DataModule

from scripts.utils import get_model, get_model_from_checkpoint, get_defaults_hyperparameters, update_sweep_configs
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from scripts.custom_callbacks import MinLossLogger, ChartsLogger
from pytorch_lightning import Trainer

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

import logging

logger = logging.getLogger("config_logger")

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path=Path(".", "Assets", "Config"), config_name="wav2vec_cls_best.yaml")  # , config_name="config.yaml")
def main(cfg: DictConfig):
    # if it's just an home test we run in offline mode
    if cfg.simulation_name == "home_test":
        os.environ["WANDB_MODE"] = "offline"

    seed_everything(0)

    wandb.init(project=cfg.simulation_name, config=get_defaults_hyperparameters(cfg))
    wandb_logger = WandbLogger(project=cfg.simulation_name)

    update_sweep_configs(hydra_cfg=cfg, sweep_cfg=wandb.config)
    logger.info(OmegaConf.to_yaml(cfg))

    # ------------------> Dataset <-----------------------
    data_module = DataModule(config=cfg)

    # ------------------> Model <-----------------------
    model = get_model(cfg)

    # saving best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./models',
        filename='checkpoint-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        save_weights_only=False
    )

    # early stopping
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=cfg.early_stopping_patience)

    # logging the best val loss
    min_val_loss_logger = MinLossLogger()

    charts_logger = ChartsLogger(classes=data_module.get_ordered_classes())

    trainer = Trainer(
        fast_dev_run=cfg.unit_test,
        logger=wandb_logger,  # W&B integration
        max_epochs=cfg.model.epochs,  # number of epochs
        callbacks=[checkpoint_callback, early_stopping_callback, min_val_loss_logger, charts_logger],
        gpus=[cfg.machine.gpu] if cfg.machine.gpu is not False else None
    )

    # ------------------> Training <-----------------------
    if cfg.train:
        trainer.fit(model=model, datamodule=data_module)

    if cfg.test:
        # if was done also the training phase use the best model just found,
        # if we are just testing without training the best model is the one specified in the config
        model_path_to_test = checkpoint_callback.best_model_path if cfg.train \
            else Path(get_original_cwd(), cfg.model_to_test)

        print("model_path_to_test: ", model_path_to_test, "\ncfg.model_to_test: ", cfg.model_to_test)

        # ------------------> Loading best model <-----------------------
        model = get_model_from_checkpoint(cfg, checkpoint_path=model_path_to_test)

        # ------------------> Testing <-----------------------
        trainer.test(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
