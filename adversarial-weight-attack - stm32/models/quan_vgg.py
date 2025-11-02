import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .quantization import *

class TinyVGG(nn.Module):
    def __init__(self, num_classes):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            # Bloc 1
            quan_Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            # Bloc 2
            quan_Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.09),                   # SpatialDropout2D(0.09) côté Keras
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            # Bloc 3
            quan_Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )

        # Global Average Pooling -> vecteur de taille 32 (nb de canaux)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifieurs denses (quantifiés)
        self.classifier = nn.Sequential(
            quan_Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            quan_Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)                # (B, 32, 1, 1)
        x = torch.flatten(x, 1)        # (B, 32)
        x = self.classifier(x)         # (B, num_classes)
        return x

def tinyvgg_quan(num_classes=10):
    """Constructs a TinyVGG model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = TinyVGG(num_classes)
    return model