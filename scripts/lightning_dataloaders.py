from hydra.utils import get_original_cwd

from pathlib import Path
from scripts.datasets.librosa_dataloaders import DEMoSDataset, RAVDESSDataset

from torch.utils.data import DataLoader, random_split
import torch
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.train, self.val, self.test = None, None, None

    def setup(self, stage=None):
        # ------------------> Dataset <-----------------------
        if self.cfg.dataset.name.lower() in ["demos", "demos_test"]:
            dataset = DEMoSDataset(root_dir=Path(get_original_cwd(), self.cfg.path.data, self.cfg.dataset.dir),
                                   padding_cropping_size=self.cfg.dataset.padding_cropping,
                                   spectrogram=self.cfg.dataset.spectrogram,
                                   sampling_rate=self.cfg.dataset.sampling_rate)
        elif self.cfg.dataset.name.lower() == "ravdess":
            dataset = RAVDESSDataset(root_dir=Path(get_original_cwd(), self.cfg.path.data, self.cfg.dataset.dir),
                                     padding_cropping_size=self.cfg.dataset.padding_cropping,
                                     spectrogram=self.cfg.dataset.spectrogram,
                                     sampling_rate=self.cfg.dataset.sampling_rate)
        else:
            raise Exception("Requested dataset, doesn't exist yet")

        # ------------------> Split < -----------------------
        self.train, self.val, self.test = random_split(dataset=dataset,
                                                       lengths=[round(len(dataset) * .8),  # train
                                                                round(len(dataset) * .1),  # val
                                                                len(dataset) - round(len(dataset) * .9)],  # test
                                                       generator=torch.Generator().manual_seed(
                                                           self.cfg.dataset.split_seed))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.machine.training_batches,
                          num_workers=self.cfg.machine.workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)
