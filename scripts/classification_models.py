import torch
import numpy as np

class SpectrogramCNN(torch.nn.Module):
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
            f = convolutions(torch.autograd.Variable(torch.ones(1,*in_size)))
            return int(np.prod(f.size()[1:]))

        linear_layer_input_size = _get_size_after_flattening(self, in_size=input_size, convolutions=self.cnn_layers)

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(linear_layer_input_size, 120),
            torch.nn.ReLU(),

            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),

            torch.nn.Linear(84, class_number),
            torch.nn.Softmax()
        )


    def forward(self, x):

        y = torch.nn.Sequential(
                self.cnn_layers,
                torch.nn.Flatten(),
                self.linear_layers
            )(x)
        
        return y
