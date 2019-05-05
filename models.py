## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()

        # input:
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(4, 4))

        # conv layers:
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=1, padding=0)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1, padding=0)

        # max-pool:
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # dropout:
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_6 = nn.Dropout(p=0.6)

        # fully connected:
        self.fc_1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc_2 = nn.Linear(in_features=1000, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=136)

        # weights initialization (as in the paper):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)
        
    def forward(self, x):
        # forward pass for conv layers, using ELU activation as suggested in the paper:
        x = self.pool(F.elu(self.conv_1(x)))
        x = self.dropout_1(x)

        x = self.pool(F.elu(self.conv_2(x)))
        x = self.dropout_2(x)

        x = self.pool(F.elu(self.conv_3(x)))
        x = self.dropout_3(x)

        x = self.pool(F.elu(self.conv_4(x)))
        x = self.dropout_4(x)

        # flattening:
        x = x.view(x.size(0), -1)

        # fully connected part with ReLU activation
        x = F.elu(self.fc_1(x))
        x = self.dropout_5(x)

        x = F.relu(self.fc_2(x))
        x = self.dropout_6(x)

        x = self.fc_3(x)

        return x
