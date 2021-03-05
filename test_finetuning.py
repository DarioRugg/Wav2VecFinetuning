"""
# Fine tuning of XLSR model for emotion recognition 
Here at first we download XLSR model; <br> Then we add the layers for the emotion classification on top on the XLSR model; <br> Finally we start with the training of this last layers.
"""

if __name__ == '__main__':

    """#### Relevant paramethers"""
    num_workers = 16
    training_batches = 32
    testing_batches = 64
    num_epoches = 8
    audio_size = 200000

    """#### Selecting the right GPU"""
    # import os
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    # os.environ["CUDA_VISIBLE_DEVICES"]="4"

    """#### Importing modules"""

    import torch
    import matplotlib.pyplot as plt
    import pandas as pd

    from scripts.classification_models import SpectrogramCNN
    from scripts.librosa_dataloaders import WavEmotionDataset

    from time import time
    import os
    from torch.utils.tensorboard import SummaryWriter

    """The paths we are going to use in the notebook"""

    data_path = os.path.join(".", "Assets", "Data")
    model_path = os.path.join(".", "Assets", "Models")
    logs_path = os.path.join(".", "Assets", "Logs")

    """## Finetuning the model

    ### Building the dataset object:
    """

    dataset = WavEmotionDataset(root_dir=os.path.join(data_path, "DEMoS_dataset"), padding_cropping_size=audio_size, specrtrogram=True)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[round(len(dataset)*0.8), len(dataset)-round(len(dataset)*0.8)], 
                                                                generator=torch.Generator().manual_seed(1234))


    """## Here the model to fine tune:"""

    number_of_classes = len(dataset.get_classes())

    model = SpectrogramCNN(input_size=(1, 128, 391), class_number=number_of_classes)



    """### Training the model"""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    logs_writer = SummaryWriter(os.path.join(logs_path, "first_try_logs"))

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
                    print(f"          - time for each observation: {round((time() - batch_beginning)/len(labels))} seconds")


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
        logs_writer.add_scalar('Accuracy', {'Validation':correct/total}, epoch)
        print(f" -> Epoch{epoch}: \n    Loss: {loss}   Accuracy: {correct/total} - epoch time: {int((time() - epoch_beginning)//60)}:{round((time() - epoch_beginning)%60)}")

    torch.save(model.state_dict(), os.path.join(model_path, f"first_try_{num_epoches}epochs.pt"))

    """### Testing now:"""
    """
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

    """