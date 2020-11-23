import torch.nn as nn


class PrintLayer(nn.Module):
    def __init__(self, name=''):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        # Do your print / debug stuff here
        # print(x)  # print(x.shape)
        print(str(x.shape) + ' ' + self.name)
        return x
