import torch
import fairseq

"""
the model below is the classificator made just with the final layers of the wav2vec model.
"""

class EmotionClassifier(torch.nn.Module):
    def __init__(self, class_number, pretrained_model=None, pretrained_path='xlsr_53_56k.pt', pretrained_out_dim=1024):

        super(EmotionClassifier, self).__init__()

        # First we take the pretrained xlsr model
        if pretrained_model is None:
            pretrained_model_list, cfg = fairseq.checkpoint_utils.load_model_ensemble([pretrained_path])
            pretrained_model = pretrained_model_list[0]
        
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()

        # then we add on top the classification layers to be trained
        self.linear_layer = torch.nn.Linear(pretrained_out_dim, class_number)
        self.softmax_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        with torch.no_grad():
            # the audio is divided in chunks depending of it's length, 
            # so we do the mean of all the chunks embeddings to get the final embedding
            embedding = self.pretrained_model(x, mask=False, features_only=True)["x"].mean(dim=1)
            
        y_pred = self.softmax_activation(self.linear_layer(embedding))
        return y_pred
