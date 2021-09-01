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
        """
        self.train, self.val, self.test = random_split(dataset=dataset,
                                                       lengths=[round(len(dataset) * .8),  # train
                                                                round(len(dataset) * .1),  # val
                                                                len(dataset) - round(len(dataset) * .8)
                                                                - round(len(dataset) * .1)],  # test
                                                       generator=torch.Generator().manual_seed(
                                                           self.cfg.dataset.split_seed))
        """

        speakers = dataset.get_speakers()
        speakers_id = speakers.unique()
        random.shuffle(speakers_id)
        print("all : ", dataset.wav_path_label_df["wav_path"].apply(lambda path: str(path)[-13:]))#.apply(lambda path: path[-13:]))
        print("val speakers: ", dataset.wav_path_label_df[speakers.isin(speakers_id[round(len(speakers_id) * .8):-round(len(speakers_id) * .1), "wav_path"].apply(lambda path: str(path)[-13:]))])
        print("val speakers: ", speakers_id[round(len(speakers_id) * .8):-round(len(speakers_id) * .1)].tolist())
        self.train = Subset(dataset, speakers.index[speakers.isin(speakers_id[:round(len(speakers_id) * .8)])])
        self.val = Subset(dataset, speakers.index[
            speakers.isin(speakers_id[round(len(speakers_id) * .8):-round(len(speakers_id) * .1)])])
        self.test = Subset(dataset, speakers.index[speakers.isin(speakers_id[-round(len(speakers_id) * .1):])])

        print("train 0: ", self.train[torch.tensor([0, 1, 2, 300, 301, 302])])
        print("val 0: ", self.val[torch.tensor([0, 1, 2, 30, 31, 32])])
        print("test 0: ", self.test[torch.tensor([0, 1, 2, 30, 31, 32])])
        print("speakers:")
        print(" train: ", self.train.get_speakers().unique())
        print(" val: ", self.val.get_speakers().unique())
        print(" test: ", self.test.get_speakers().unique())

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.machine.training_batches,
                          num_workers=self.cfg.machine.workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.machine.training_batches, num_workers=self.cfg.machine.workers)
