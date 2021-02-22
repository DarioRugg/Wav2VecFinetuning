# steel ned to work on it!
"""
# Fine tuning of XLSR model for emotion recognition 
Here at first we download XLSR model; <br> Then we add the layers for the emotion classification on top on the XLSR model; <br> Finally we start with the training of this last layers.
"""

"""#### Relevant paramethers"""
num_workers = 16
testing_batches = 64
num_epoches = 8
audio_size = 200000

"""#### Selecting the right GPU"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="4"

"""#### Importing modules"""

import torch
import torchaudio 
import fairseq
from fairseq.models.wav2vec import Wav2VecModel
import matplotlib.pyplot as plt
import pandas as pd

from scripts.classification_models import EmotionClassifier
from scripts.dataloaders import WavEmotionDataset

from time import time
from torch.utils.tensorboard import SummaryWriter

"""The paths we are going to use in the notebook"""

data_path = "./Assets/Data"
model_path = "./Assets/Models"
logs_path = "./Logs"

"""### Logs writer:"""

logs_writer = SummaryWriter(os.path.join(logs_path, "first_try_logs"))

criterion = torch.nn.CrossEntropyLoss()

"""## Getting the XLSR model."""

xlsr_model_list, cfg = fairseq.checkpoint_utils.load_model_ensemble([os.path.join(model_path, 'xlsr_53_56k.pt')], )
xlsr_pretrained = xlsr_model_list[0]

"""## Here the model to fine tune:"""

number_of_classes = 8

model = EmotionClassifier(class_number=8, pretrained_model=xlsr_pretrained)
model.load_state_dict(torch.load(os.path.join(model_path, f"first_try_{num_epoches}epochs.pt")))
model.eval()

"""### Testing now:"""

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=testing_batches, num_workers=num_workers)
correct = 0
total = 0
loss = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss += criterion(outputs, labels)/len(val_loader)


logs_writer.add_scalar('Accuracy', {'Test':correct/total}, num_epoches)
print(f" ---> On Test data: \n    Loss: {loss}   Accuracy: {correct/total}")

logs_writer.close()