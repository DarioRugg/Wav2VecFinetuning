import torch
from transformers import Wav2Vec2Model

"""
the model below is the classificator made just with the final layers of the wav2vec model.
"""

class Wav2VecClassifier(torch.nn.Module):
    def __init__(self, num_classes, pretrained_out_dim=1024):

        super(Wav2VecClassifier, self).__init__()

        # First we take the pretrained xlsr model        
        self.pretrained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.pretrained_model.eval()

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, num_classes)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        with torch.no_grad():
            # the audio is divided in chunks depending of it's length, 
            # so we do the mean of all the chunks embeddings to get the final embedding
            embedding = self.pretrained_model(x, mask=False, features_only=True)["x"].mean(dim=1)
            embedding = self.pretrained_model( x ).last_hidden_state.mean(dim=1)
            
        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred
