import torch
from .quantization import *

class Mul(torch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
            #torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
            #             stride=stride, padding=padding, groups=groups, bias=False)
            quan_Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
    )

def construct_model(num_class):
    num_class = num_class
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.Dropout2d(p=0.2, inplace=False),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        torch.nn.Dropout2d(p=0.2, inplace=False),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.Dropout2d(p=0.2, inplace=False),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.Dropout2d(p=0.2, inplace=False),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        quan_Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last).cuda()
    return model

def resnet9_quan(num_class=10):
  return construct_model(num_class=num_class)
    

