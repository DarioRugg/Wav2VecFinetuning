import torch
import numpy as np
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

from collections import OrderedDict


class BaseLightningModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super(BaseLightningModel, self).__init__()

        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = Accuracy()(y_hat.to("cpu"), y)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = Accuracy()(y_hat.to("cpu"), y)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # test_step defined the test loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss, on_epoch=True)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = Accuracy()(y_hat.to("cpu"), y)
        self.log('test_acc', acc, on_epoch=True)
        return {"y": y, "y_hat": y_hat}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)


class SpectrogramCNN(BaseLightningModel):
    def __init__(self, input_size, class_number, learning_rate,
                 cnn_hidden_layers=4, cnn_filters=64,
                 classifier_hidden_layers=2, classifier_hidden_size=16,
                 drop_out_prob=0.03):

        super(SpectrogramCNN, self).__init__(learning_rate)

        self.cnn_layers = torch.nn.Sequential(OrderedDict([
            ("input_layer", torch.nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=2)),
            ("input_activation", torch.nn.ReLU())
        ]))
        for i in range(cnn_hidden_layers):
            self.cnn_layers.add_module(f"hidden_{i + 1}",
                                       torch.nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, stride=2))
            self.cnn_layers.add_module(f"activation_{i + 1}", torch.nn.ReLU())
            if i % 2 == 0:
                self.cnn_layers.add_module(f"dropout_{i + 1}", torch.nn.Dropout(p=drop_out_prob))

        self.cnn_layers.add_module(f"last_hidden_layer", torch.nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters*2, kernel_size=3, stride=2))
        self.cnn_layers.add_module(f"last_hidden_activation", torch.nn.ReLU())
        self.cnn_layers.add_module(f"last_hidden_dropout", torch.nn.Dropout(p=drop_out_prob))

        self.cnn_layers.add_module(f"output_layer", torch.nn.Conv2d(in_channels=cnn_filters*2, out_channels=cnn_filters*2, kernel_size=3, stride=2))
        self.cnn_layers.add_module(f"output_activation", torch.nn.ReLU())

        def _get_size_after_flattening(in_size, convolutions):
            f = convolutions(torch.autograd.Variable(torch.ones(1, *in_size)))
            return int(np.prod(f.size()[1:]))

        linear_layer_input_size = _get_size_after_flattening(in_size=input_size, convolutions=self.cnn_layers)

        self.classifier = torch.nn.Sequential(OrderedDict([
            ("input_layer", torch.nn.Linear(linear_layer_input_size, classifier_hidden_size)),
            ("input_activation", torch.nn.ReLU())
        ]))
        for i in range(classifier_hidden_layers):
            self.classifier.add_module(f"hidden_{i + 1}", torch.nn.Linear(classifier_hidden_size, classifier_hidden_size))
            self.classifier.add_module(f"activation_{i + 1}", torch.nn.ReLU())
            if i % 2 == 0:
                self.cnn_layers.add_module(f"dropout_{i + 1}", torch.nn.Dropout(p=drop_out_prob))

        self.classifier.add_module(f"output_layer", torch.nn.Linear(classifier_hidden_size, class_number))

        self.model = torch.nn.Sequential(
            self.cnn_layers,
            torch.nn.Flatten(),
            self.classifier
        )


class EfficientNetModel(BaseLightningModel):
    def __init__(self, num_classes, blocks, learning_rate):
        super(EfficientNetModel, self).__init__(learning_rate)

        self.model = EfficientNet.from_pretrained(model_name=f"efficientnet-b{blocks}", in_channels=1,
                                                  num_classes=num_classes)
