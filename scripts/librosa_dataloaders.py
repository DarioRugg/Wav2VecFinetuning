import torch
import librosa
import pandas as pd
import numpy as np
import os

"""
the dataloader below is for loading the DEMoS dataset.
"""

class WavEmotionDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, classes_dict=None, padding_cropping_size=None, specrtrogram=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the DEMoS audio files.
            classes_dict (dict): dictionary class label -> meaning, if None default classes are used.
            padding_cropping_size (int): size to crop and pad, in order to have all wavs of the same length.
            spectrogram (bool): if True gets out the spectrogram instead of the raw sampling
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        demos_dir = os.path.join(root_dir, "DEMOS")
        neu_dir = os.path.join(root_dir, "NEU")

        if classes_dict is None:
            classes_dict = {"col": "Guilt",
                        "dis": "Disgust",
                        "gio": "Happiness",
                        "pau": "Fear",
                        "rab": "Anger",
                        "sor": "Surprise",
                        "tri": "Sadness",
                        "neu": "Neutral"}

        self.classes = sorted(list(classes_dict.keys()))
        self.classes_dict = classes_dict
        self.padding_cropping_size = padding_cropping_size
        self.get_spectrogram = specrtrogram
        self.transform = transform

        paths = list(map(lambda fname: os.path.join(demos_dir, fname), sorted(os.listdir(demos_dir)))) + list(map(lambda fname: os.path.join(neu_dir, fname), sorted(os.listdir(neu_dir))))
        labels = list(map(lambda fname: self.classes.index(fname.split("_")[-1][:3]), sorted(os.listdir(demos_dir)))) + list(map(lambda fname: self.classes.index(fname.split("_")[-1][:3]), sorted(os.listdir(neu_dir))))

        self.wav_path_label_df = pd.DataFrame({"wav_path": paths, "label": labels})

    def __len__(self):
        return len(self.wav_path_label_df)

    def get_classes(self):
        return self.classes

    def __getitem__(self, idx):

        def _get_data_from_file(audio_path, padd_crop_size=None, get_spectrogrm=False):

            def _padding_cropping(input_wav, size):
                if len(input_wav) > size:
                    input_wav = input_wav[:size]
                elif len(input_wav) < size:
                    input_wav = np.pad(input_wav, pad_width=(0, size-len(input_wav)), constant_values=0)
                return input_wav

            y, sr = librosa.load(audio_path, sr=None)

            if padd_crop_size is not None:
                y = _padding_cropping(y, padd_crop_size)

            if get_spectrogrm:
                y = librosa.feature.melspectrogram(y=y, sr=sr)

            return torch.unsqueeze(torch.tensor(y), 0)

        if torch.is_tensor(idx):
            idx = idx.tolist()

            X = torch.stack(list(map(lambda audio_path: _get_data_from_file(audio_path, padd_crop_size=self.padding_cropping_size, get_spectrogrm=self.get_spectrogram), self.wav_path_label_df.iloc[idx, 0])))
            y = torch.tensor(self.wav_path_label_df.iloc[idx, 1].tolist())

        else:
            X = _get_data_from_file(self.wav_path_label_df.iloc[idx, 0], padd_crop_size=self.padding_cropping_size, get_spectrogrm=self.get_spectrogram)
            y = self.wav_path_label_df.iloc[idx, 1]

        if self.transform:
            X = self.transform(X)

        return X, y
    
    def _get_data_from_file(audio_path):
        return torch.tensor(librosa.load(audio_path, sr=None)[0])
