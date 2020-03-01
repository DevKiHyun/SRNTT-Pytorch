import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn


class SRNTT(nn.Module):
    def __init__(self, n_channels, n_res_blocks):
        super(SRNTT, self).__init__()
        
        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels
        self.vgg_channels = [256, 128, 64] # n_channels of VGG19 - `relu3_1', `relu2_1`, `relu1_1' 
        
        """ Resnet 1 for input iamge """
        self.module_1 = nn.ModuleList([
            nn.Conv2d(3, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[ResBlock(in_channels=self.n_channels, out_channels=self.n_channels,
                       name=f'ResBlock1_{i}') for i in range(self.n_res_blocks)],
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels)]
        )
        
        """ Resnet 2 & Sub-pixel """
        self.module_2 = nn.Sequential(
            nn.Conv2d(self.n_channels+self.vgg_channels[0], self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[ResBlock(in_channels=self.n_channels, out_channels=self.n_channels,
                       name=f'ResBlock2_{i}') for i in range(self.n_res_blocks)],
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels)
        )
        
        self.module_2_upscale = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        
        """ Resnet 3 & Sub-pixel """
        self.module_3 = nn.Sequential(
            nn.Conv2d(self.n_channels+self.vgg_channels[1], self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[ResBlock(in_channels=self.n_channels, out_channels=self.n_channels,
                       name=f'ResBlock3_{i}') for i in range(self.n_res_blocks//2)],
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels)
        )
        
        self.module_3_upscale = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
            
        
        """ Resnet 4 """
        self.module_4 = nn.Sequential(
            nn.Conv2d(self.n_channels+self.vgg_channels[2], self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[ResBlock(in_channels=self.n_channels, out_channels=self.n_channels,
                       name=f'ResBlock4_{i}') for i in range(self.n_res_blocks//4)],
            nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels)
        )
        
        """ Last Step """
        self.reconstruct = nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_channels//2, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x, maps):
        """ ResBlock 1 """
        skip_connection = None
        for i, layer in enumerate(self.module_1):
            x = layer(x)
            if i == 1: # output of Conv(#1), Relu(#1) 
                skip_connection = x

        x += skip_connection # add(#1 + #18)
        skip_connection = x
        
        x = torch.cat((x, maps['relu3_1']), dim=1) 
        
        
        """ ResBlock 2 """
        x = self.module_2(x) + skip_connection
        x = self.module_2_upscale(x)
        skip_connection = x
        
        x = torch.cat((x, maps['relu2_1']), dim=1) 
        
        
        """ ResBlock 3 """
        x = self.module_3(x) + skip_connection
        x = self.module_3_upscale(x)
        skip_connection = x
        
        x = torch.cat((x, maps['relu1_1']), dim=1) 

        """ ResBlock 4 """
        x = self.module_4(x) + skip_connection

        """ Reconstruct """
        x = self.reconstruct(x)
        x = torch.clamp(x, min=0, max=1) # input range is [0,1]
        
        return x 
        
                                   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, name=None):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
                                   
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
                      
        self.model = nn.Sequential(OrderedDict([
            (f'{self.name}_conv1', self.conv1),
            (f'{self.name}_bn1',   nn.BatchNorm2d(self.out_channels)),
            (f'{self.name}_relu1', nn.ReLU(inplace=True)),
            (f'{self.name}_conv2', self.conv2),
            (f'{self.name}_bn2',   nn.BatchNorm2d(self.out_channels))
        ]))
       
        
    def forward(self, x):
        identity = x
        
        out = self.model(x)
        out += identity
        
        return out
