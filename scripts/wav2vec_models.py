import torch
from transformers import Wav2Vec2Model

"""
the model below is the classificator made just with the final layers of the wav2vec model.
"""

class Wav2VecComplete(torch.nn.Module):
    def __init__(self, num_classes, pretrained_out_dim=1024, finetune_pretrained=False):

        super(Wav2VecComplete, self).__init__()
        self.finetune_pretrained = finetune_pretrained

        # First we take the pretrained xlsr model        
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

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
    
    def train(self):
        # we train the pretrained architecture only if specified
        if self.finetune_pretrained: self.pretrained_model.train()
        else: self.pretrained_model.eval()

        self.linear_layer.train()


class Wav2VecFeatureExtractor(torch.nn.Module):
    def __init__(self, num_classes, pretrained_out_dim=512, finetune_pretrained=False):

        super(Wav2VecFeatureExtractor, self).__init__()
        self.finetune_pretrained = finetune_pretrained
        
        # First we take the pretrained xlsr model        
        complete_pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        self.pretrained_model = complete_pretrained_model.feature_extractor

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
    
    def train(self):
        # we train the pretrained architecture only if specified
        if self.finetune_pretrained: self.pretrained_model.train()
        else: self.pretrained_model.eval()

        self.linear_layer.train()


class Wav2VecFeezingEncoderOnly(torch.nn.Module):
    def __init__(self, num_classes, pretrained_out_dim=1024):

        super(Wav2VecFeezingEncoderOnly, self).__init__()

        # First we take the pretrained xlsr model        
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        
        embedding = self.pretrained_model( x ).last_hidden_state.mean(dim=1)
            
        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred

    def train(self):
        # we don't want to train the encoder as well
        self.pretrained_model.encoder.eval()

        self.pretrained_model.feature_extractor.train()
        self.pretrained_model.feature_projection.train()
        
        self.linear_layer.train()