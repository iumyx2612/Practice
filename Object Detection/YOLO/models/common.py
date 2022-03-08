import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.act(x)

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.ReLU, **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, **kwargs)
        self.act = act()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x
