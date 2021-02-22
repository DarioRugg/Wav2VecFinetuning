import torch
import torchaudio 
import pandas as pd
import os

"""
the dataloader below is for loading the DEMoS dataset.
"""

class WavEmotionDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, classes_dict=None, padding_cropping_size=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the DEMoS audio files.
            classes_dict (dict): dictionary class label -> meaning.
            padding_cropping_size (int): size to crop and pad, in order to have all wavs of the same length.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        demos_dir = os.path.join(root_dir, "DEMOS")
        neu_dir = os.path.join(root_dir, "NEU")

        self.classes = sorted(list(classes_dict.keys()))
        self.classes_dict = classes_dict
        self.transform = transform
        self.padding_cropping_size = padding_cropping_size

        paths = list(map(lambda fname: os.path.join(demos_dir, fname), sorted(os.listdir(demos_dir)))) + list(map(lambda fname: os.path.join(neu_dir, fname), sorted(os.listdir(neu_dir))))
        labels = list(map(lambda fname: self.classes.index(fname.split("_")[-1][:3]), sorted(os.listdir(demos_dir)))) + list(map(lambda fname: self.classes.index(fname.split("_")[-1][:3]), sorted(os.listdir(neu_dir))))

        self.wav_path_label_df = pd.DataFrame({"wav_path": paths, "label": labels})

    def __len__(self):
        return len(self.wav_path_label_df)

    def __getitem__(self, idx):

        def _padding_cropping(input_wav, size):
            if len(input_wav) > size:
                input_wav = input_wav[:size]
            elif len(input_wav) < size:
                input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
            return input_wav

        if torch.is_tensor(idx):
            idx = idx.tolist()

            X = torch.stack(list(map(lambda audio_path: _padding_cropping(torch.squeeze(torchaudio.load(audio_path)[0], dim=0), self.padding_cropping_size) if self.padding_cropping_size is not None 
                                     else torch.squeeze(torchaudio.load(audio_path)[0], dim=0), self.wav_path_label_df.iloc[idx, 0])))
            y = torch.tensor(self.wav_path_label_df.iloc[idx, 1].tolist())

        else:
            X = _padding_cropping(torch.squeeze(torchaudio.load(self.wav_path_label_df.iloc[idx, 0])[0], dim=0), self.padding_cropping_size) if self.padding_cropping_size is not None else torch.squeeze(torchaudio.load(self.wav_path_label_df.iloc[idx, 0])[0], dim=0)
            y = self.wav_path_label_df.iloc[idx, 1]

        if self.transform:
            X = self.transform(X)

        return X, y