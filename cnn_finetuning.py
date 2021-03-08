"""
# Fine tuning of XLSR model for emotion recognition 
Here at first we download XLSR model; <br> Then we add the layers for the emotion classification on top on the XLSR model; <br> Finally we start with the training of this last layers.
"""
"""#### Importing modules"""

import torch

from scripts.classification_models import SpectrogramCNN
from scripts.librosa_dataloaders import WavEmotionDataset

from time import time
from os.path import join
import os
from torch.utils.tensorboard import SummaryWriter
import json

if __name__ == '__main__':

    """The paths we are going to use in the notebook"""
    data_path = join(".", "Assets", "Data")
    model_path = join(".", "Assets", "Models")
    logs_path = join(".", "Assets", "Logs")
    conf_path = join(".", "Assets", "Configs")

    # ------> Loading the proper config <-------
    conf_file = "server_config.json"
    with open(join(conf_path, conf_file)) as f:
        conf = json.load(f)

    """#### Relevant paramethers"""
    simulation_name = conf["simulation_name"]
    num_workers = conf["num_workers"]
    training_batches = conf["training_batches"]
    testing_batches = conf["testing_batches"]
    num_epoches = conf["num_epoches"]
    audio_size = conf["audio_size"]
    split_seed = conf["train_test_split_seed"]

    """#### Selecting the right GPU"""
    if "GPU infos" in conf.keys():
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]="4"


    """## Finetuning the model

    ### Building the dataset object:
    """

    dataset = WavEmotionDataset(root_dir=os.path.join(data_path, "DEMoS_dataset"), padding_cropping_size=audio_size, specrtrogram=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[round(len(dataset)*0.8), len(dataset)-round(len(dataset)*0.8)], 
                                                                generator=torch.Generator().manual_seed(split_seed))


    """## Here the model to fine tune:"""

    number_of_classes = len(dataset.get_classes())

    model = SpectrogramCNN(input_size=(1, 128, 391), class_number=number_of_classes)



    """### Training the model"""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    logs_writer = SummaryWriter(os.path.join(logs_path, f"{simulation_name}_logs"))

    best_model = {"dict": None, "epoch": None, "loss": None, "accuracy": None}
    for epoch in range(num_epoches):
        print(f" -> Starting epoch {epoch} <- ")
        epoch_beginning = time()

        train_split, val_split = torch.utils.data.random_split(dataset=dataset, lengths=[round(len(dataset)*0.8), len(dataset)-round(len(dataset)*0.8)])
        train_loader = torch.utils.data.DataLoader(train_split, batch_size=training_batches, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(train_split, batch_size=testing_batches, num_workers=num_workers)

        for i, batch in enumerate(train_loader):
            batch_beginning = time()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % (len(train_loader)//10) == 0:
                print(f"        Batch {i}, {round(i/len(train_loader)*100)}%; Loss: {loss}")
                if epoch == 0 and i == 0:
                    print(f"          - time for each observation: {round((time() - batch_beginning)/len(labels), 3)} seconds")
                    print(f"            in total should take around {round((time() - batch_beginning)/len(labels)*len(train_dataset), 3)} seconds")


        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels)/len(val_loader)

        logs_writer.add_scalars('Loss', {'Train':loss,'Validation':val_loss}, epoch)
        logs_writer.add_scalars('Accuracy', {'Validation':correct/total}, epoch)
        print(f" -> Epoch{epoch}: \n    Loss: {loss}   Accuracy: {correct/total} - epoch time: {int((time() - epoch_beginning)//60)}:{round((time() - epoch_beginning)%60)}")

        if best_model is None or loss < best_model["loss"]:
            best_model = {"State_Dict": model.state_dict(), "Epoch": epoch, "Loss": loss, "Accuracy": correct/total}

    # ----> saving models at the end of the trainging <------
    torch.save(model.state_dict(), os.path.join(model_path, f"{simulation_name}_best_model.pt")) # best model
    torch.save(model.state_dict(), os.path.join(model_path, f"{simulation_name}_last_model_{num_epoches}epochs.pt")) # last epoch model
