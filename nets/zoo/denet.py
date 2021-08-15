import torch
import torch.nn as nn
import skimage
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class _downsampling(nn.Module):
    def __init__(self, channel_in):
        super(_downsampling, self).__init__()        
        #channel_in, channel_out = channel_var

        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        
        self.bn = nn.BatchNorm2d(2*channel_in)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1= self.conv(x)
        out2= self.maxpool(x)
        
        out = torch.cat([out1, out2], 1)
        out = self.relu(self.bn(out))
        return out
    

class _linear_residual(nn.Module):
    def __init__(self, channel_in):
        super(_linear_residual, self).__init__()        
        #channel_in, channel_out = channel_var

        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(int(channel_in/4.))
        self.relu1= nn.ELU(alpha=1.673)
        
        self.conv2 = nn.Conv2d(in_channels=int(channel_in/4.), out_channels=int(channel_in/4.), kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(channel_in/4.))
        self.relu2= nn.ELU(alpha=1.673)
        
        self.conv3 = nn.Conv2d(in_channels=int(channel_in/4.), out_channels=channel_in, kernel_size=1, stride=1, padding=0)        
        
    def forward(self, x):
        residual = x        
        _lambda = 1.051
        
        out = self.bn1(self.conv1(x))
        out = self.relu1(out) * _lambda
        
        out = self.bn2(self.conv2(out))
        out = self.relu2(out) * _lambda
        
        out = self.conv3(out)
        
        out = torch.add(out, residual)
        return out
    
class _encoding_block(nn.Module):
    def __init__(self, channel_in):
        super(_encoding_block, self).__init__()

        self.block_1 = nn.Sequential(
            _linear_residual(channel_in=channel_in),
            _linear_residual(channel_in=channel_in),
            _linear_residual(channel_in=channel_in),
            _linear_residual(channel_in=channel_in),
            _linear_residual(channel_in=channel_in),
            _linear_residual(channel_in=channel_in),
        )
        
    def forward(self, x):
        return self.block_1(x)
        

class _compressing_module(nn.Module):
    def __init__(self, channel_in):
        super(_compressing_module, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(int(channel_in/4.))
        self.relu1= nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=int(channel_in/4.), out_channels=int(channel_in/4.), kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(int(channel_in/4.))
        self.relu2= nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=int(channel_in/4.), out_channels=channel_in, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        residual = x
        
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        
        out = self.bn2(self.conv2(out))
        out = self.relu2(out)
        
        out = self.conv3(out)
        return out
        

class _duc(nn.Module):
    def __init__(self):
        super(_duc, self).__init__()

        self.subpixel = nn.PixelShuffle(8)
        
    def forward(self, x):
        #out = self.relu(self.conv(x))
        #out = self.subpixel(out)
        out = self.subpixel(x)
        return out

    
class DeNet(nn.Module):
    def __init__(self, pretrained=False,mode='Train'):
        super(DeNet, self).__init__()
        self.mode=mode
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        #self.relu1 = nn.PReLU()
        self.relu1 = nn.ReLU()
        #self.relu1 = nn.LeakyReLU(0.1)
        self.DS_block_1 = self.make_layer(_downsampling, 64)
        self.EC_block_1 = self.make_layer(_encoding_block, 128)
        
        self.DS_block_2 = self.make_layer(_downsampling, 128)
        self.EC_block_2 = self.make_layer(_encoding_block, 256)
        
        self.DS_block_3 = self.make_layer(_downsampling, 256)
        self.EC_block_3 = self.make_layer(_encoding_block, 512)
        
        self.CP_block_41= self.make_layer(_compressing_module, 512)
        self.EC_block_42= self.make_layer(_encoding_block, 512)
        self.CP_block_43= self.make_layer(_compressing_module, 512)
        self.EC_block_44= self.make_layer(_encoding_block, 512)
        self.CP_block_45= self.make_layer(_compressing_module, 512)
        self.EC_block_46= self.make_layer(_encoding_block, 512)
        self.CP_block_47= self.make_layer(_compressing_module, 512)
        
        self.conv_f = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0)
        #self.relu2 = nn.PReLU()
        self.relu2 = nn.ReLU()
        #self.relu2 = nn.LeakyReLU(0.1)
        
        self.dcu = _duc()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv_i(x))
        out = self.DS_block_1(out)
        out = self.EC_block_1(out)
        
        out = self.DS_block_2(out)
        out = self.EC_block_2(out)
        
        out = self.DS_block_3(out)
        out = self.EC_block_3(out)
        
        out = self.CP_block_41(out)
        out = self.EC_block_42(out)
        out = self.CP_block_43(out)
        out = self.EC_block_44(out)
        out = self.CP_block_45(out)
        out = self.EC_block_46(out)
        out = self.CP_block_47(out)
        
        out = self.relu2(self.conv_f(out))
        out = self.dcu(out)
        if self.mode == 'Train':
            return F.sigmoid(out)
        elif self.mode == 'Infer':
            return out