import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator(nn.Module):
    """------------------------------- """
    """# | Layer name(s) | Output size """
    """------------------------------- """
    """0 |     Input     |  160x160x3  """
    """1 | Conv,BN,LReLU |  160x160x32 """
    """2 | Conv,BN,LReLU |   80x80x32  """
    """3 | Conv,BN,LReLU |   80x80x64  """
    """4 | Conv,BN,LReLU |   40x40x64  """
    """5 | Conv,BN,LReLU |   40x40x128 """
    """6 | Conv,BN,LReLU |   20x20x128 """
    """7 | Conv,BN,LReLU |   20x20x256 """
    """8 | Conv,BN,LReLU |   10x10x256 """
    """9 | Conv,BN,LReLU |   10x10x512 """
    """10| Conv,BN,LReLU |    5x5x512  """
    """11|    Flatten    |     12800   """
    """12|   FC, LReLU   |     1024    """
    """13|       FC      |       1     """
    def __init__(self, negative_slope=0.2):
        super(Discriminator, self).__init__()
        
        self.module = nn.Sequential(
            *self.sub_module(3, 32), # 160x160x32
            *self.sub_module(32, 32, stride=2), #80x80x32
            *self.sub_module(32, 64), #80x80x64
            *self.sub_module(64, 64, stride=2), #40x40x64
            *self.sub_module(64, 128), #40x40x128
            *self.sub_module(128, 128, stride=2), #20x20x128
            *self.sub_module(128, 256), #20x20x256
            *self.sub_module(256, 256, stride=2), #10x10x256
            *self.sub_module(256, 512), #10x10x512
            *self.sub_module(512, 512, stride=2),
            Flatten(), # 5x5x512 --> 12800
            nn.Linear(12800, 1024),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
        
    def sub_module(self, input_channels, output_channels, kernel_size=3, stride=1, negative_slope=0.2):
        nn_list = [
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        ]
        
        return nn_list
                      
    def forward(self, x):
        return self.module(x)