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

"""
class SpectrogramCNN(torch.nn.Module):
    def __init__(self, class_number):

        super(EmotionClassifier, self).__init__()

        self.cnn_layers = Sequential(
            self.conv1 = torch.nn.Conv2d(3, 6, 5),
            self.pool = torch.nn.MaxPool2d(2, 2),
            self.conv2 = torch.nn.Conv2d(6, 16, 5)
        )

        self.linear_layers = Sequential(
            self.fc1 = torch.nn.Linear(16 * 5 * 5, 120),
            self.fc2 = torch.nn.Linear(120, 84),
            self.fc3 = torch.nn.Linear(84, class_number)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
"""