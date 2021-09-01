import torch
import numpy as np
from torch.nn.functional import cross_entropy
from pytorch_lightning.metrics.functional import accuracy
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet


class BaseLightningModel(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_acc', acc, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_epoch=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y_hat, y)
        self.log('test_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer


class SpectrogramCNN(BaseLightningModel):
    def __init__(self, input_size, class_number):
        super(SpectrogramCNN, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            torch.nn.ReLU(),
        )

        def _get_size_after_flattening(self, in_size, convolutions):
            f = convolutions(torch.autograd.Variable(torch.ones(1, *in_size)))
            return int(np.prod(f.size()[1:]))

        linear_layer_input_size = _get_size_after_flattening(self, in_size=input_size, convolutions=self.cnn_layers)

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(linear_layer_input_size, 120),
            torch.nn.ReLU(),

            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),

            torch.nn.Linear(84, class_number)
        )

    def forward(self, x):
        y = torch.nn.Sequential(
            self.cnn_layers,
            torch.nn.Flatten(),
            self.linear_layers
        )(x)
        return y


class EfficientNetModel(BaseLightningModel):
    def __init__(self, num_classes, blocks, learning_rate):
        super(EfficientNetModel, self).__init__()

        self.lr = learning_rate

        self.model = EfficientNet.from_pretrained(model_name=f"efficientnet-b{blocks}", in_channels=1,
                                                  num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)
