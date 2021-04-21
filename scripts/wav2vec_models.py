import torch
from torch.nn.functional import cross_entropy
import pytorch_lightning as pl
from transformers import Wav2Vec2Model
from scripts.wav2vec_cls_model import Wav2VecModelOverridden


class Wav2VecBase(pl.LightningModule):
    def __init__(self, num_classes, pretrained_out_dim=1024):

        super(Wav2VecBase, self).__init__()

        # First we take the pretrained xlsr model
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Wav2VecCLSToken(Wav2VecBase):

    def __init__(self, num_classes, pretrained_out_dim=1024):
        super(Wav2VecCLSToken, self).__init__(num_classes, pretrained_out_dim)

        # We replace the pretrained model with the one with the CLS token
        self.pretrained_model = Wav2VecModelOverridden.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # require grad for all the model:
        for name, param in self.pretrained_model.named_parameters():
            param.requires_grad = True
        # then freezing the encoder only, except for the normalization layers that we want to fine-tune:
        for name, param in self.pretrained_model.encoder.named_parameters():
            if "layer_norm" not in name:
                param.requires_grad = False

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
            embedding = self.pretrained_model( x ).last_hidden_state.mean(dim=1)
            
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
        if self.finetune_pretrained: self.pretrained_model.train()
        else: self.pretrained_model.eval()

        self.linear_layer.train()
    
    def eval(self):
        self.pretrained_model.eval()
        self.linear_layer.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Wav2VecFeatureExtractor(pl.LightningModule):
    def __init__(self, num_classes, pretrained_out_dim=512, finetune_pretrained=False):

        super(Wav2VecFeatureExtractor, self).__init__()
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
            embedding = self.pretrained_model( x ).mean(dim=2)
            
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
        if self.finetune_pretrained: self.pretrained_model.train()
        else: self.pretrained_model.eval()

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
        
        embedding = self.pretrained_model( x ).last_hidden_state.mean(dim=1)
            
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
