import torch
from torch import nn
from torch.nn.functional import cross_entropy
import pytorch_lightning as pl
from transformers import Wav2Vec2Model, Wav2Vec2Config
from scripts.models.wav2vec2_modified import Wav2VecModelOverridden


class Wav2VecBase(pl.LightningModule):
    def __init__(self, num_classes):
        super(Wav2VecBase, self).__init__()

        # First we take the pretrained xlsr model
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        pretrained_out_dim = self.pretrained_model.config.hidden_size

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        embedding = self.pretrained_model(x).last_hidden_state.mean(dim=1)
        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer


class Wav2VecFeatureExtractor(Wav2VecBase):
    def __init__(self, num_classes, pretrained_out_dim=(512, 226), finetune_pretrained=True):
        super(Wav2VecFeatureExtractor, self).__init__(num_classes=num_classes)
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.pretrained_model = complete_pretrained_model.feature_extractor

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        # then we add on top the classification layers to be trained
        self.linear_projector = nn.Sequential(
            # nn.Linear(torch.prod(torch.tensor(pretrained_out_dim)), num_classes),
            nn.Linear(30720, num_classes),
            nn.Softmax(dim=0)
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


class Wav2VecFeatureExtractorGAP(Wav2VecBase):
    def __init__(self, num_classes, finetune_pretrained=True):
        super(Wav2VecFeatureExtractorGAP, self).__init__(num_classes=num_classes)
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.pretrained_model = complete_pretrained_model.feature_extractor

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        self.cls_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        with torch.enable_grad() if self.finetune_pretrained else torch.no_grad():
            # the features are like a one channel image
            features = self.pretrained_model(x)

        # we need to add the first channel to the "image"
        features = torch.unsqueeze(features, dim=1)
        # we feed this image in the cls_net that gives the classification tensor
        y_pred = self.cls_net(features)
        return y_pred


class Wav2VecCLSToken(Wav2VecBase):

    def __init__(self, num_classes):
        super(Wav2VecCLSToken, self).__init__(num_classes)

        # We replace the pretrained model with the one with the CLS token
        self.pretrained_model = Wav2VecModelOverridden.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # we don't want to get the masks
        self.pretrained_model.config.mask_time_prob = 0

        # require grad for all the model:
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True
        """
        # then freezing the encoder only, except for the normalization layers that we want to fine-tune:
        for name, param in self.pretrained_model.encoder.named_parameters():
            if "layer_norm" not in name:
                param.requires_grad = False
        """

    def forward(self, x):
        cls_token, _ = self.pretrained_model(x)

        y_pred = self.softmax_activation(self.linear_layer(cls_token))
        return y_pred


class Wav2VecCLSTokenNotPretrained(Wav2VecBase):

    def __init__(self, num_classes):
        super(Wav2VecCLSTokenNotPretrained, self).__init__(num_classes)

        # getting the config for constructing the model randomly initialized
        model_config = Wav2Vec2Config("facebook/wav2vec2-large-xlsr-53")

        # we don't want to get the masks
        model_config.mask_time_prob = 0

        # We replace the pretrained model with a non pretrained architecture with CLS token
        self.pretrained_model = Wav2VecModelOverridden(model_config)
        pretrained_out_dim = self.pretrained_model.config.hidden_size

        # the output dim is changed
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)

        # require grad for all the model:
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True
        # """
        # then freezing the encoder only, except for the normalization layers that we want to fine-tune:
        for name, param in self.pretrained_model.encoder.named_parameters():
            if "layer_norm" not in name:
                param.requires_grad = False
        # """

    def forward(self, x):

        cls_token, _ = self.pretrained_model(x)

        y_pred = self.softmax_activation(self.linear_layer(cls_token))
        return y_pred


class Wav2VecComplete(pl.LightningModule):
    def __init__(self, num_classes, pretrained_out_dim=1024, finetune_pretrained=False):

        super(Wav2VecComplete, self).__init__()
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        with torch.enable_grad() if self.finetune_pretrained else torch.no_grad():
            # the audio is divided in chunks depending of it's length, 
            # so we do the mean of all the chunks embeddings to get the final embedding
            embedding = self.pretrained_model(x).last_hidden_state.mean(dim=1)

        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def train(self):
        # we train the pretrained architecture only if specified
        if self.finetune_pretrained:
            self.pretrained_model.train()
        else:
            self.pretrained_model.eval()

        self.linear_layer.train()

    def eval(self):
        self.pretrained_model.eval()
        self.linear_layer.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Wav2VecFeatureExtractor_old(pl.LightningModule):
    def __init__(self, num_classes, pretrained_out_dim=512, finetune_pretrained=False):

        super(Wav2VecFeatureExtractor_old, self).__init__()
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model        
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.pretrained_model = complete_pretrained_model.feature_extractor

        # setting require grad = true only if we want to fine tune the pretrained model
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = self.finetune_pretrained

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        with torch.enable_grad() if self.finetune_pretrained else torch.no_grad():
            # now we have as dimensions: [batch, featuremap, audiochunks]
            # so we average on the dim 2
            embedding = self.pretrained_model(x).mean(dim=2)

        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def train(self):
        # we train the pretrained architecture only if specified
        if self.finetune_pretrained:
            self.pretrained_model.train()
        else:
            self.pretrained_model.eval()

        self.linear_layer.train()

    def eval(self):
        self.pretrained_model.eval()
        self.linear_layer.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Wav2VecFeezingEncoderOnly(pl.LightningModule):
    def __init__(self, num_classes, pretrained_out_dim=1024):

        super(Wav2VecFeezingEncoderOnly, self).__init__()

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
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):

        embedding = self.pretrained_model(x).last_hidden_state.mean(dim=1)

        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def train(self):
        # we don't want to train the encoder as well
        self.pretrained_model.encoder.eval()

        self.pretrained_model.feature_extractor.train()
        self.pretrained_model.feature_projection.train()

        self.linear_layer.train()

    def eval(self):
        self.pretrained_model.eval()
        self.linear_layer.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
