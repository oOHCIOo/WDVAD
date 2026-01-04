import os
import torch
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))
        alpha = torch.tanh(self.classifier(x))
        if is_training:
            x = self.dropout(x)
        x = self.sigmoid(self.classifier(x))
        x=alpha*(torch.pow(x,2)-x)+x

        return x
