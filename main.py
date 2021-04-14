import hydra
from omegaconf import DictConfig

from scripts.utils import server_setup
from scripts.classification_models import SpectrogramCNN

import torch

from scripts.utils import get_model, get_dataset, split_dataset, get_model_from_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint

from time import time
from os.path import join

from pytorch_lightning.loggers import WandbLogger

# Pytorch-Lightning
from pytorch_lightning import Trainer


@hydra.main(config_path=r"Assets\Config", config_name="config.yaml")
def main(cfg: DictConfig):
    server_setup(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandbLogger(project=cfg.simulation_name, save_dir=join(hydra.utils.get_original_cwd(), "wandb_logs"))

    # ------------------> Dataset <-----------------------
    train_dataset, test_dataset = get_dataset(cfg)

    train_split, val_split = split_dataset(train_dataset, split_size=0.8, seed=None)
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=cfg.machine.training_batches,
                                               num_workers=cfg.machine.workers)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=cfg.machine.testing_batches,
                                             num_workers=cfg.machine.workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.machine.testing_batches,
                                              num_workers=cfg.machine.workers)

    # ------------------> Model <-----------------------
    model = get_model(cfg)

    model = model.to(device)

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
        max_epochs=cfg.model.epoches,  # number of epochs
        callbacks=[checkpoint_callback]
    )

    # ------------------> Training <-----------------------
    if cfg.train:
        trainer.fit(model, train_loader, val_loader)

    if cfg.test:
        # ------------------> Loading best model <-----------------------
        model = get_model_from_checkpoint(cfg, checkpoint_path=checkpoint_callback.best_model_path)

        # ------------------> Testing <-----------------------
        trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
