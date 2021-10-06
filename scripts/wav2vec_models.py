from typing import Optional, Callable
import itertools
from collections import OrderedDict

import torch
from torch import nn
from torch.optim import Optimizer
from transformers import Wav2Vec2Model, Wav2Vec2Config
from scripts.models.wav2vec2_modified import Wav2VecModelOverridden

from scripts.classification_models import BaseLightningModel


class Wav2VecCLSPaperFinetuning(BaseLightningModel):

    def __init__(self, num_classes, learning_rate, num_epochs, hidden_layers=0, hidden_size=None, drop_out_prob=0.03):
        super(Wav2VecCLSPaperFinetuning, self).__init__(learning_rate)

        self.lr = learning_rate
        self.num_epochs = num_epochs

        # We replace the pretrained model with the one with the CLS token
        self.pretrained_model = Wav2VecModelOverridden.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # freezing the feature extractor (we are not going to finetune it)
        for name, param in self.pretrained_model.feature_extractor.named_parameters():
            param.requires_grad = False

        # then we add on top the classification layer to be trained
        if hidden_layers == 0:
            self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, num_classes)
        else:

            self.classifier = nn.Sequential(OrderedDict([
                ("input_layer", nn.Linear(self.pretrained_model.config.hidden_size, hidden_size)),
                ("input_activation", nn.ReLU())
            ]))
            for i in range(hidden_layers - 1):
                self.classifier.add_module(f"hidden_{i + 1}", nn.Linear(hidden_size, hidden_size))
                self.classifier.add_module(f"activation_{i + 1}", nn.ReLU())
                if i % 2 == 0:
                    self.classifier.add_module(f"dropout_{i + 1}", nn.Dropout(p=drop_out_prob))

            self.classifier.add_module(f"output_layer", nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        cls_token, _ = self.pretrained_model(x)

        y_pred = self.classifier(cls_token)
        return y_pred

    # here we must define the optimizer and the different learning rate
    def configure_optimizers(self):
        optimizer_linear_layer = torch.optim.Adam(params=self.classifier.parameters(), lr=self.lr)

        params = [self.pretrained_model.feature_projection.parameters(),
                  self.pretrained_model.encoder.parameters(),
                  self.classifier.parameters()]
        optimizer_linear_and_encoder = torch.optim.Adam(
            # params=itertools.chain(*params),
            params=itertools.chain(*params),
            lr=self.lr)
        return optimizer_linear_layer, optimizer_linear_and_encoder

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:

        # for the first 30% of updates we train only the linear layer
        # for the rest of the updates the encoder gets finetuned as well
        if (0.3 >= epoch / self.num_epochs and optimizer_idx == 0) or \
                (0.3 < epoch / self.num_epochs and optimizer_idx == 1):

            # warm-up for the first 10%
            if epoch < self.num_epochs // 10:
                lr_scale = min(1., float(epoch + 1) / float(self.num_epochs // 10))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * optimizer.defaults["lr"]
            # constant learning rate for the next 40%
            # linearly decaying for the final 50%
            elif epoch >= self.num_epochs // 2:
                lr_scale = min(1.,
                               1 - (float(epoch - self.num_epochs // 2) / float(
                                   self.num_epochs - self.num_epochs // 2)))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * optimizer.defaults["lr"]

            # update params
            optimizer.step(closure=optimizer_closure)

    # override training_step for adding the optimizer_idx parameter.
    def training_step(self, batch, batch_idx, optimizer_idx):
        return super(Wav2VecCLSPaperFinetuning, self).training_step(batch, batch_idx)


class Wav2VecFeatureExtractor(BaseLightningModel):
    def __init__(self, num_classes, learning_rate, pretrained_out_dim=(512, 226), finetune_pretrained=True):
        super(Wav2VecFeatureExtractor, self).__init__(learning_rate)
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.pretrained_model = complete_pretrained_model.feature_extractor

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        # then we add on top the classification layers to be trained
        self.linear_projector = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(pretrained_out_dim)).item(), num_classes)
        )

    def forward(self, x):
        with torch.enable_grad() if self.finetune_pretrained else torch.no_grad():
            # the features are like a spectrogram, an image with one channel
            features = self.pretrained_model(x)

        # first we flatten everything
        features = torch.flatten(features, start_dim=1)
        # then we use the linear projection for prediction
        y_pred = self.linear_projector(features)
        return y_pred


class Wav2VecFeatureExtractorGAP(BaseLightningModel):
    def __init__(self, num_classes, learning_rate, finetune_pretrained=True, cnn_hidden_layers=2, cnn_filters=16,
                 drop_out_prob=0.05):
        super(Wav2VecFeatureExtractorGAP, self).__init__(learning_rate)
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.pretrained_model = complete_pretrained_model.feature_extractor

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        self.cnn_layers = nn.Sequential(OrderedDict([
            ("input_layer", nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=2)),
            ("input_activation", nn.ReLU())
        ]))
        # for i in range(cnn_hidden_layers):
        #     self.cnn_layers.add_module(f"hidden_{i + 1}",
        #                                nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3,
        #                                          stride=2))
        #     self.cnn_layers.add_module(f"activation_{i + 1}", nn.ReLU())
        #     if i % 2 == 0:
        #         self.cnn_layers.add_module(f"dropout_{i + 1}", nn.Dropout(p=drop_out_prob))
        #
        self.cnn_layers.add_module("output_layer",
                                   nn.Conv2d(in_channels=cnn_filters, out_channels=num_classes, kernel_size=3,
                                             stride=2))
        self.cnn_layers.add_module("global_average_pooling", nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        with torch.enable_grad() if self.finetune_pretrained else torch.no_grad():
            # the features are like a one channel image
            features = self.pretrained_model(x)


        print("tokens", features.shape)
        print("after unsqueeze", torch.unsqueeze(features, dim=1).shape)
        # we need to add the first channel to the "image"
        features = self.cnn_layers(torch.unsqueeze(features, dim=1))
        print("before", features.shape)
        # we feed this image in the cnn_layers that gives the classification tensor
        y_pred = torch.reshape(features, shape=(features.shape[0], features.shape[1]))
        print("after", features.shape)
        return y_pred


class Wav2VecCLSToken(BaseLightningModel):

    def __init__(self, num_classes, learning_rate):
        super(Wav2VecCLSToken, self).__init__(learning_rate)

        # We replace the pretrained model with the one with the CLS token
        self.pretrained_model = Wav2VecModelOverridden.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # we don't want to get the masks
        # self.pretrained_model.config.mask_time_prob = 0

        # require grad for all the model:
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True
        """
        # then freezing the encoder only, except for the normalization layers that we want to fine-tune:
        for name, param in self.pretrained_model.encoder.named_parameters():
            if "layer_norm" not in name:
                param.requires_grad = False
        """

        pretrained_out_dim = self.pretrained_model.config.hidden_size
        # then we add on top the classification layer to be trained
        self.linear_layer = nn.Linear(pretrained_out_dim, num_classes)

    def forward(self, x):
        cls_token, _ = self.pretrained_model(x)

        y_pred = self.linear_layer(cls_token)
        return y_pred


class Wav2VecCLSTokenNotPretrained(BaseLightningModel):

    def __init__(self, num_classes, learning_rate):
        super(Wav2VecCLSTokenNotPretrained, self).__init__(learning_rate)

        # getting the config for constructing the model randomly initialized
        model_config = Wav2Vec2Config("facebook/wav2vec2-large-xlsr-53")

        # we don't want to get the masks
        model_config.mask_time_prob = 0

        # We replace the pretrained model with a non pretrained architecture with CLS token
        self.pretrained_model = Wav2VecModelOverridden(model_config)

        pretrained_out_dim = self.pretrained_model.config.hidden_size
        # then we add on top the classification layer to be trained
        self.linear_layer = nn.Linear(pretrained_out_dim, num_classes)

    def forward(self, x):
        cls_token, _ = self.pretrained_model(x)

        y_pred = self.softmax_activation(self.linear_layer(cls_token))
        return y_pred


class Wav2VecComplete(BaseLightningModel):
    def __init__(self, num_classes, learning_rate, pretrained_out_dim=1024):
        super(Wav2VecComplete, self).__init__(learning_rate)

        # First we take the pretrained xlsr model
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True

        # then we add on top the classification layers to be trained
        self.linear_layer = nn.Linear(pretrained_out_dim, num_classes)

    def forward(self, x):
        # the audio is divided in chunks depending of it's length,
        # so we do the mean of all the chunks embeddings to get the final embedding
        embedding = self.pretrained_model(x).last_hidden_state.mean(dim=1)

        y_pred = self.linear_layer(embedding)
        return y_pred


class Wav2VecFeezingEncoderOnly(BaseLightningModel):
    def __init__(self, num_classes, learning_rate, pretrained_out_dim=1024):

        super(Wav2VecFeezingEncoderOnly, self).__init__(learning_rate)

        # First we take the pretrained xlsr model        
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # require grad for all the model:
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True
        # then freezing the encoder only, except for the normalization layers that we want to fine-tune:
        for name, param in self.pretrained_model.encoder.named_parameters():
            if "layer_norm" not in name:
                param.requires_grad = False

        # then we add on top the classification layers to be trained
        self.linear_layer = nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = nn.Softmax(dim=0)

    def forward(self, x):

        embedding = self.pretrained_model(x).last_hidden_state.mean(dim=1)

        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred
