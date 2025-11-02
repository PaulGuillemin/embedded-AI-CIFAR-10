import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .quantization import *


class cnn_bones(nn.Module):

    def __init__(self, num_classes):
        super(cnn_bones, self).__init__()
        self.conv1 = quan_Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = quan_Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = quan_Linear(2752, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = quan_Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

        
def cnn_quan(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = cnn_bones(num_classes)
    return model