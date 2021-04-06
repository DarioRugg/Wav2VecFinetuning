
import torch
import hydra

from scripts.librosa_dataloaders import DEMoSDataset, RAVDESSDataset

from os.path import join
import os

from scripts.classification_models import SpectrogramCNN
from scripts.wav2vec_models import Wav2VecComplete, Wav2VecFeatureExtractor, Wav2VecFeezingEncoderOnly
from efficientnet_pytorch import EfficientNet



def get_dataset(cfg, split=True, part="both"):

    # ------------------> Dataset <-----------------------
    orig_cwd = hydra.utils.get_original_cwd()
    if cfg.dataset.name.lower() in ["demos", "demos_test"]:
        dataset = DEMoSDataset(root_dir=join(orig_cwd, cfg.path.data, cfg.dataset.dir), padding_cropping_size=cfg.dataset.padding_cropping, spectrogram=cfg.dataset.spectrogram, sampling_rate=cfg.dataset.sampling_rate)
    elif cfg.dataset.name.lower() == "ravdess":
        dataset = RAVDESSDataset(root_dir=join(orig_cwd, cfg.path.data, cfg.dataset.dir), padding_cropping_size=cfg.dataset.padding_cropping, spectrogram=cfg.dataset.spectrogram, sampling_rate=cfg.dataset.sampling_rate)

    if not split: return dataset

    # ------------------> Split <-----------------------
    train_dataset, test_dataset = split_dataset(dataset, cfg.dataset.split_size, cfg.dataset.split_seed)
    
    if part is None or part == "both": return train_dataset, test_dataset
    elif part == "test": return test_dataset
    elif part == "train": return train_dataset

def split_dataset(dataset, split_size, seed):
    # ------------------> Split <-----------------------
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[round(len(dataset)*split_size), len(dataset)-round(len(dataset)*split_size)], 
                                                                generator=torch.Generator().manual_seed(seed) if seed is not None else None)
    
    return train_dataset, test_dataset


def get_model(cfg):
    if cfg.model.name.lower() == "cnn":
        return SpectrogramCNN(input_size=(1, 128, 391), class_number=cfg.dataset.number_of_classes)
    elif cfg.model.name.lower() == "efficientnet":
        return EfficientNet.from_pretrained(model_name=f"efficientnet-b{cfg.model.blocks}", in_channels=1, num_classes=cfg.dataset.number_of_classes)
    elif cfg.model.name.lower() == "wav2vec":
        if cfg.model.option == "all" & cfg.model.finetuning == "partial":
            return Wav2VecFeezingEncoderOnly(num_classes=cfg.dataset.number_of_classes)
        elif cfg.model.option == "all":
            return Wav2VecFeatureExtractor(num_classes=cfg.dataset.number_of_classes, finetune_pretrained=cfg.model.finetuning)
        elif cfg.model.option == "cnn":
            return Wav2VecComplete(num_classes=cfg.dataset.number_of_classes, finetune_pretrained=cfg.model.finetuning)

def server_setup(cfg):
    if cfg.machine.gpu is not False:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.machine.gpu)
