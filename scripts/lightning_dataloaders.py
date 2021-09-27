from hydra.utils import get_original_cwd

from pathlib import Path
from scripts.datasets.librosa_dataloaders import DEMoSDataset, RAVDESSDataset

from torch.utils.data import DataLoader, random_split, Subset
import torch
import pytorch_lightning as pl

import random

random.seed(1234)


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        # fetching the ordered class names of the dataset
        self.ordered_class_names = self._get_dataset().get_ordered_classes()

        self.train, self.val, self.test = None, None, None

    def _get_dataset(self):
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
        return dataset

    def setup(self, stage=None):
        # ------------------> Dataset <-----------------------
        dataset = self._get_dataset()

        # ------------------> Split < -----------------------
        if self.cfg.dataset.speaker_split:
            # -----> By Speaker <------
            speakers = dataset.get_speakers()
            speakers_id = speakers.unique()
            random.shuffle(speakers_id)
            self.train = Subset(dataset, speakers.index[speakers.isin(speakers_id[:round(len(speakers_id) * .8)])])
            self.val = Subset(dataset, speakers.index[
                speakers.isin(speakers_id[round(len(speakers_id) * .8):-round(len(speakers_id) * .1)])])
            self.test = Subset(dataset, speakers.index[speakers.isin(speakers_id[-round(len(speakers_id) * .1):])])
        else:
            # -----> Random <------
            self.train, self.val, self.test = random_split(dataset=dataset,
                                                           lengths=[round(len(dataset) * .8),  # train
                                                                    round(len(dataset) * .1),  # val
                                                                    len(dataset) - round(len(dataset) * .8)
                                                                    - round(len(dataset) * .1)],  # test
                                                           generator=torch.Generator().manual_seed(
                                                               self.cfg.dataset.split_seed))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.machine.training_batches,
                          num_workers=self.cfg.machine.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)

    def get_ordered_classes(self):
        return self.ordered_class_names
