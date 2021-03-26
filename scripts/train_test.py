"""#### Importing modules"""

import torch

from scripts.librosa_dataloaders import DEMoSDataset, RAVDESSDataset

from time import time
from os.path import join
import os
from torch.utils.tensorboard import SummaryWriter
import json

from scripts.classification_models import SpectrogramCNN
from scripts.wav2vec_models import Wav2VecComplete, Wav2VecFeatureExtractor, Wav2VecFeezingEncoderOnly
from efficientnet_pytorch import EfficientNet


"""The paths we are going to use in the notebook"""
data_path = join(".", "Assets", "Data")
model_path = join(".", "Assets", "Models")
logs_path = join(".", "Assets", "Logs")
conf_path = join(".", "Assets", "Configs")


def train(conf_file):    

    # ------------------> Loading the proper config <-----------------------
    with open(join(conf_path, conf_file)) as f:
        conf = json.load(f)

    # general:
    simulation_name = conf["simulation_name"]
    num_epoches = conf["num_epoches"]

    # model:
    model_name = conf["model"]
    model_architecture = conf["model_arch"] if "model_arch" in conf.keys() else None
    wav2vec_finetuning = conf["finetuning_flag"] if "finetuning_flag" in conf.keys() else True

    # dataset:
    dataset_name = conf["dataset"]
    audio_size = conf["audio_size"]
    spectrogram_flag = conf["use_spectrogram"]
    sampling_rate = conf["sampling_rate"] if "sampling_rate" in conf.keys() else None
    train_split_size = conf["train_test_split_size"]
    split_seed = conf["train_test_split_seed"]

    # hardware:
    num_workers = conf["num_workers"]
    training_batches = conf["training_batches"]
    testing_batches = conf["testing_batches"]
    if "server_config" in conf.keys(): _server_setup(conf["server_config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # ------------------> Dataset <-----------------------
    dataset = _get_dataset(dataset=dataset_name, pad_crop_size=audio_size, specrtrogram=spectrogram_flag, sampling_rate=sampling_rate)

    train_dataset = _get_dataset_split(data=dataset, part="train", split_size=train_split_size, seed=split_seed)

    
    # ------------------> Model <-----------------------
    number_of_classes = len(dataset.get_classes())
    
    model = _get_model(model_name=model_name, num_classes=number_of_classes, model_arch=model_architecture, wav2vec_finetuning_flag=wav2vec_finetuning)

    model = model.to(device)
    
    # ------------------> Training <-----------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    logs_writer = SummaryWriter(os.path.join(logs_path, f"{simulation_name}_logs"))

    best_model = None
    for epoch in range(num_epoches):
        print(f" -> Starting epoch {epoch} <- ")
        epoch_beginning = time()

        train_split, val_split = _get_dataset_split(data=train_dataset, part="both", split_size=0.8, seed=None)
        train_loader = torch.utils.data.DataLoader(train_split, batch_size=training_batches, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size=testing_batches, num_workers=num_workers)

        # in train model for training
        model.train()

        train_loss = 0
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
               
            if i % (len(train_loader)//10) == 0:
                print(f"        Batch {i}, {round(i/len(train_loader)*100)}%; Loss: {loss}")
                if epoch == 0:
                    total_time = (time() - epoch_beginning)/labels.size(0)*len(train_dataset)*num_epoches
                    print(f"          - In total it should take around {round(total_time/60)} minutes")
            train_loss += loss.item()/len(train_loader)

        # model eval before validation round 
        model.eval()

        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels)/len(val_loader)

        logs_writer.add_scalars('Loss', {'Train':train_loss,'Validation':val_loss}, epoch)
        logs_writer.add_scalars('Accuracy', {'Validation':correct/total}, epoch)
        print(f"    Epoch {epoch} summary: \n    Loss: {val_loss}   Accuracy: {correct/total} - epoch time: {int((time() - epoch_beginning)//60)}:{int((time() - epoch_beginning)%60)}")
        

        if best_model is None or val_loss < best_model["Loss"]:
            best_model = {"State_Dict": model.state_dict(), "Epoch": epoch, "Loss": val_loss.item(), "Accuracy": correct/total}


    # ----> saving models at the end of the trainging <------
    print(f"Saving models, best model found at epoch {best_model['Epoch']} with Loss {round(best_model['Loss'], 4)} and Accuracy {round(best_model['Accuracy'], 4)} on val")

    torch.save(best_model["State_Dict"], os.path.join(model_path, f"{simulation_name}_best_model.pt")) # best model
    torch.save(model.state_dict(), os.path.join(model_path, f"{simulation_name}_last_model_{num_epoches}epochs.pt")) # last epoch model




def test(conf_file):
    
    # ------------------> Loading configuration <-----------------------
    with open(join(conf_path, conf_file)) as f:
        conf = json.load(f)

    # general:
    simulation_name = conf["simulation_name"]

    # model:
    model_name = conf["model"]
    model_architecture = conf["model_arch"] if "model_arch" in conf.keys() else None
    wav2vec_finetuning = conf["finetuning_flag"] if "finetuning_flag" in conf.keys() else True

    # dataset:
    dataset_name = conf["dataset"]
    audio_size = conf["audio_size"]
    spectrogram_flag = conf["use_spectrogram"]
    sampling_rate = conf["sampling_rate"] if "sampling_rate" in conf.keys() else None
    split_seed = conf["train_test_split_seed"]
    train_split_size = conf["train_test_split_size"]

    # hardware:
    num_workers = conf["num_workers"]
    training_batches = conf["training_batches"]
    testing_batches = conf["testing_batches"]
    num_epoches = conf["num_epoches"]
    if "server_config" in conf.keys(): _server_setup(conf["server_config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    # ------------------> Dataset <-----------------------

    dataset = _get_dataset(dataset=dataset_name, pad_crop_size=audio_size, specrtrogram=spectrogram_flag, sampling_rate=sampling_rate)

    test_dataset = _get_dataset_split(data=dataset, part="test", split_size=train_split_size, seed=split_seed)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)

    
    # ------------------> Model <-----------------------

    number_of_classes = len(dataset.get_classes())

    model = _get_model(model_name=model_name, num_classes=number_of_classes, model_arch=model_architecture, wav2vec_finetuning_flag=wav2vec_finetuning)

    # we just wnat to test the model
    model.eval()

    model = model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, f"{simulation_name}_best_model.pt")))

    logs_writer = SummaryWriter(os.path.join(logs_path, f"{simulation_name}_logs"))

    
    # ------------------> Testing <-----------------------

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)
    correct = 0
    total = 0
    loss = 0
    test_start_time = time()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += torch.nn.CrossEntropyLoss()(outputs, labels)/len(test_loader)


    logs_writer.add_scalars('Accuracy', {'Test':correct/total}, num_epoches)
    print(f" ---> On Test data: \n    Loss: {loss}   Accuracy: {correct/total}")
    print(f"      done in: {round(time()-test_start_time, 2)} seconds")

    logs_writer.close()



# ------------------> Other functions <-----------------------
def _get_model(model_name, num_classes, model_arch=None, wav2vec_finetuning_flag=False):
    if model_name.lower() == "cnn":
        return SpectrogramCNN(input_size=(1, 128, 391), class_number=num_classes)
    elif model_name.lower() == "effnet":
        return EfficientNet.from_pretrained(model_name=model_arch, in_channels=1, num_classes=num_classes)
    elif model_name.lower() == "wav2vec":
        if model_arch == "complete":
            return Wav2VecComplete(num_classes=num_classes, finetune_pretrained=wav2vec_finetuning_flag)
        elif model_arch == "cnn_only":
            return Wav2VecFeatureExtractor(num_classes=num_classes, finetune_pretrained=wav2vec_finetuning_flag)
        elif model_arch == "finetuning_convs_frozen_encoder":
            return Wav2VecFeezingEncoderOnly(num_classes=num_classes)

def _get_dataset(dataset, pad_crop_size, specrtrogram=False, sampling_rate=None):
    if dataset.lower() == "demos":
        return DEMoSDataset(root_dir=os.path.join(data_path, "DEMoS_dataset"), padding_cropping_size=pad_crop_size, specrtrogram=specrtrogram, sampling_rate=sampling_rate)
    elif dataset.lower() == "ravdess":
        return RAVDESSDataset(root_dir=os.path.join(data_path, "RAVDESS_dataset"), padding_cropping_size=pad_crop_size, specrtrogram=specrtrogram, sampling_rate=sampling_rate)
    elif dataset.lower() == "demos_short_test":
        return DEMoSDataset(root_dir=os.path.join(data_path, "DEMoS_dataset_short_test"), padding_cropping_size=pad_crop_size, specrtrogram=specrtrogram, sampling_rate=sampling_rate)

def _get_dataset_split(data, part, split_size, seed):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset=data, lengths=[round(len(data)*split_size), len(data)-round(len(data)*split_size)], 
                                                                generator=torch.Generator().manual_seed(seed) if seed is not None else None)
    
    if part is None or part == "both": return train_dataset, test_dataset
    elif part == "test": return test_dataset
    elif part == "train": return train_dataset

def _server_setup(server_config_file):
    with open(join(conf_path, server_config_file)) as f:
        conf = json.load(f)

    os.environ["CUDA_DEVICE_ORDER"]=conf["CUDA_DEVICE_ORDER"]  
    os.environ["CUDA_VISIBLE_DEVICES"]=conf["CUDA_VISIBLE_DEVICES"]

