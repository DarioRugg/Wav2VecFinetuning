"""#### Importing modules"""

import torch

from scripts.utils import get_model, get_dataset, split_dataset
import hydra

from time import time
from os.path import join

import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


def train(cfg, tensorboard_writer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger = WandbLogger(project='MNIST', save_dir=hydra.utils.get_original_cwd())

    # ------------------> Dataset <-----------------------
    train_dataset = get_dataset(cfg, part="train")
    
    # ------------------> Model <-----------------------
    model = get_model(cfg)

    model = model.to(device)
    
    # ------------------> Training <-----------------------

    print(model)

    train_split, val_split = split_dataset(train_dataset, split_size=0.8, seed=None)
    train_loader = torch.utils.data.DataLoader(train_split, batch_size=cfg.machine.training_batches, num_workers=cfg.machine.workers)
    val_loader = torch.utils.data.DataLoader(val_split, batch_size=cfg.machine.testing_batches, num_workers=cfg.machine.workers)
    
    trainer = Trainer(
        logger=wandb_logger,    # W&B integration
        max_epochs=cfg.model.epoches            # number of epochs
        )

    trainer.fit(model, train_loader, val_loader)

    print("Done")

    """
    best_model = None
    for epoch in range(cfg.model.epoches):
        print(f" -> Starting epoch {epoch} <- ")
        epoch_beginning = time()

        train_split, val_split = split_dataset(train_dataset, split_size=0.8, seed=None)
        train_loader = torch.utils.data.DataLoader(train_split, batch_size=cfg.machine.training_batches, num_workers=cfg.machine.workers)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size=cfg.machine.testing_batches, num_workers=cfg.machine.workers)

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
                if epoch == 0 and i == 0:
                    total_time = (time() - epoch_beginning)/labels.size(0)*len(train_dataset)*cfg.model.epoches
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

        tensorboard_writer.add_scalars('Loss', {'Train':train_loss,'Validation':val_loss}, epoch)
        tensorboard_writer.add_scalars('Accuracy', {'Validation':correct/total}, epoch)
        wandb.log({"Val_Loss": val_loss, "Train_Loss": train_loss})
        wandb.log({"Val_Accuracy": correct/total})

        print(f"    Epoch {epoch} summary: \n    Loss: {val_loss}   Accuracy: {correct/total} - epoch time: {int((time() - epoch_beginning)//60)}:{int((time() - epoch_beginning)%60)}")
        

        if best_model is None or val_loss < best_model["Loss"]:
            best_model = {"State_Dict": model.state_dict(), "Epoch": epoch, "Loss": val_loss.item(), "Accuracy": correct/total}
            wandb.run.summary["best_loss_train"] = val_loss.item()
            wandb.run.summary["best_accuracy_train"] = correct/total
            wandb.run.summary["best_at_epoch_train"] = epoch


    # ----> saving models at the end of the trainging <------
    print(f"Saving models, best model found at epoch {best_model['Epoch']} with Loss {round(best_model['Loss'], 4)} and Accuracy {round(best_model['Accuracy'], 4)} on val")

    torch.save(best_model["State_Dict"], join("models", f"best_model.pt")) # best model
    torch.save(model.state_dict(), join("models", f"last_model_epoch{epoch}.pt")) # last epoch model
    """




def test(cfg, tensorboard_writer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------> Dataset <-----------------------
    test_dataset = get_dataset(cfg, part="test")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.machine.testing_batches, num_workers=cfg.machine.workers)
    
    # ------------------> Model <-----------------------

    model = get_model(cfg=cfg)

    # we just wnat to test the model
    model.eval()

    model = model.to(device)

    model.load_state_dict(torch.load(join("models", f"best_model.pt")))

    
    # ------------------> Testing <-----------------------
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


    tensorboard_writer.add_scalars('Accuracy', {'Test':correct/total}, cfg.model.epoches)
    wandb.run.summary["best_loss_test"] = loss.item()
    wandb.run.summary["best_accuracy_test"] = correct/total

    print(f" ---> On Test data: \n    Loss: {loss}   Accuracy: {correct/total}")
    print(f"      done in: {round(time()-test_start_time, 2)} seconds")
