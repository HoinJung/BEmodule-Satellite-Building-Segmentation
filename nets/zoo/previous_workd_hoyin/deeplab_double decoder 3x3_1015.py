import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models

    
# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from .aspp import ASPP, ASPP_Bottleneck


class _up_piexl(nn.Module):
    def __init__(self, channel_in):
        super(_up_piexl, self).__init__()
        self.relu = nn.ReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.subpixel(out)
        return out
    
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class _up_bilinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_up_bilinear, self).__init__()
        self.middle_channels = (in_channels + out_channels) //2
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, self.middle_channels),
            ConvRelu(self.middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)    

class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()

        self.num_classes = 1
        self.resnet = Make_Resnet(num_filters)        
        self.aspp = ASPP(num_filters * 16 , num_filters * 64) 
        self.sub_up3 = _up_piexl(num_filters * 64)
        self.sub_up2 = _up_piexl(num_filters * 32)
        self.sub_up1 = _up_piexl(num_filters * 16)
        self.bi_up3 = _up_bilinear(num_filters * 64, num_filters * 16)
        self.bi_up2 = _up_bilinear(num_filters * 32, num_filters * 8)
        self.bi_up1 = _up_bilinear(num_filters * 16, num_filters * 4)
        self.conv3 = conv_block(num_filters * 32)
        self.conv2 = conv_block(num_filters * 16)
        self.conv1 = conv_block(num_filters * 8)

        self.conv_f = nn.Conv2d(in_channels=num_filters*16, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)
     

    def forward(self, x):
        ###encoder 
        temp_ret = self.resnet(x) 
        enc1, enc2, enc3, enc4 = temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3]

        out = self.aspp(enc4)                   # 1024
        ##decoder, bilinear
        out_bi = self.bi_up3(out)               # [batchsize, 256, 128x128]
        out_bi = torch.cat([enc3, out_bi],1)    # [batchsize, 512, 128x128]
        out_bi = self.conv3(out_bi)
        
        out_bi = self.bi_up2(out_bi)            # [batchsize, 128, 256x256]
        out_bi = torch.cat([enc2, out_bi],1)
        out_bi = self.conv2(out_bi)             # [batchsize, 256, 256x256]
        
        out_bi = self.bi_up1(out_bi)            # [batchsize, 64, 512x512]
        out_bi = torch.cat([enc1, out_bi],1)    # [batchsize, 128, 512x512]
        out_bi = self.conv1(out_bi) 
               
        ##decoder, subpixel                     
        out_sub = self.sub_up3(out)               # [batchsize, 256, 128x128]
        out_sub = torch.cat([enc3, out_sub],1)    # [batchsize, 512, 128x128]
        out_sub = self.conv3(out_sub)
        
        out_sub = self.sub_up2(out_sub)            # [batchsize, 128, 256x256]
        out_sub = torch.cat([enc2, out_sub],1)    # [batchsize, 256, 256x256]
        out_sub = self.conv2(out_sub)
        
        out_sub = self.sub_up1(out_sub)            # [batchsize, 64, 512x512]
        out_sub = torch.cat([enc1, out_sub],1)    # [batchsize, 128, 512x512]
        out_sub = self.conv1(out_sub)
        
        ##final output
        concat_f = torch.cat([out_sub, out_bi],1)
        result = self.conv_f(concat_f)

        
        return result

  
    
#################################Resnet#######################################    
    
class conv_block(nn.Module):


    def __init__(self, in_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv(x)))
        out = self.bn(self.conv(out)) 
        out = out + residual
        out = F.relu(out) 

        return out

class Make_Resnet(nn.Module):
    def __init__(self, in_channel):
        super(Make_Resnet, self).__init__()
        
        self.in_channel = 64
        num_blocks_layer_2 = 2    
        num_blocks_layer_3 = 2
        num_blocks_layer_4 = 2
        num_blocks_layer_5 = 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,  bias=False)        #entrance
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = make_layer(BasicBlock, in_channels=64, channels=64, num_blocks=num_blocks_layer_2, stride=1, dilation=1)   #64ch 1/4
        self.layer3 = make_layer(BasicBlock, in_channels=64, channels=128, num_blocks=num_blocks_layer_3, stride=2, dilation=1)  #128ch 1/8
        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=2, dilation=1)  #256ch 1/16
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=256, num_blocks=num_blocks_layer_5, stride=2, dilation=1)  # 512ch 1/32
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  
        
        #out = self.maxpool()
        c1 = self.layer2(out)
        c2 = self.layer3(c1)
        c3 = self.layer4(c2) 
        c4 = self.layer5(c3) 
        
        #return [c1, c2, c3, c4, c5]
        return [c1, c2, c3, c4]

    
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) 

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class Bottleneck_2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck_2, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out
