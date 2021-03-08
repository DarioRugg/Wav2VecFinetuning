
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
    

    dataset = WavEmotionDataset(root_dir=os.path.join(data_path, "DEMoS_dataset"), padding_cropping_size=audio_size, specrtrogram=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[round(len(dataset)*0.8), len(dataset)-round(len(dataset)*0.8)], 
                                                                generator=torch.Generator().manual_seed(split_seed))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)


    """## Here the best model:"""

    number_of_classes = len(dataset.get_classes())

    model = SpectrogramCNN(input_size=(1, 128, 391), class_number=number_of_classes)

    model.load_state_dict(torch.load(os.path.join(model_path, f"{simulation_name}_best_model.pt")))

    logs_writer = SummaryWriter(os.path.join(logs_path, f"{simulation_name}_logs"))

    """### Testing now:"""
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)
    correct = 0
    total = 0
    loss = 0
    test_start_time = time()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += torch.nn.CrossEntropyLoss()(outputs, labels)/len(test_loader)


    logs_writer.add_scalars('Accuracy', {'Test':correct/total}, num_epoches)
    print(f" ---> On Test data: \n    Loss: {loss}   Accuracy: {correct/total}")
    print(f"      done in: {round(time()-test_start_time, 2)} seconds")

    logs_writer.close()