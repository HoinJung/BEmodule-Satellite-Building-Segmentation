import torch
import torch.nn as nn
import skimage
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class _up_deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_up_deconv, self).__init__()        
        
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.deconv(x)
        out = self.relu(out)
        
        return out

class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.conv(out)
        return out
    
class _up(nn.Module):
    def __init__(self, channel_in,factor):
        super(_up, self).__init__()
#         self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(factor)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
#         out = self.relu(self.conv(x))
        out = self.conv(x)
        out = self.subpixel(out)
        return out

    
class hed_unet_shuffle(nn.Module):

    def __init__(self, n_class=1, pretrained=False):
        super().__init__()
        
        n_channels = 32
                

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.dconv_down5 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)                
        self.upsample5 = _up_deconv(1024,512)        
        self.upsample4 = _up_deconv(512,256)
        self.upsample3 = _up_deconv(256,128)
        self.upsample2 = _up_deconv(128,64)
        self.dconv_up4 = double_conv(512 + 512, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)
        
                
        self.side_up2 = _up(128,2)
        self.side_up3 = _up(256,4)
        self.side_up4 = _up(512,8)
        self.side_up5 = _up(1024,16)
        
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(32, 1, 1)
        self.dsn3 = nn.Conv2d(16, 1, 1)
        self.dsn4 = nn.Conv2d(8, 1, 1)
        self.dsn5 = nn.Conv2d(4, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)        
        self.final = nn.Conv2d(69,1,1)

        self.relu = nn.ReLU()        


        # modify
        self.dconv_fuse_boundary = double_conv(69, 69)
        self.dconv_fuse_mask = double_conv(64+69,64+69)
        self.final_boundary = nn.Conv2d(69,1,1)
        self.final_mask = nn.Conv2d(64+69,1,1)

        
    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)        
        conv5 = self.dconv_down5(x)        
        x = self.upsample5(conv5)        
        x = torch.cat([x, conv4],1)        
        x = self.dconv_up4(x)
        x = self.upsample4(x) 
        x = torch.cat([x, conv3], dim=1)        
        x = self.dconv_up3(x)
        x = self.upsample3(x)        
        x = torch.cat([x, conv2], dim=1)  
        x = self.dconv_up2(x)
        x = self.upsample2(x)        
        x = torch.cat([x, conv1], dim=1)           
        x = self.dconv_up1(x)
        
        ## side output
        d1 = self.dsn1(conv1)
        d2 = self.dsn2(self.side_up2(conv2))
        d3 = self.dsn3(self.side_up3(conv3))
        d4 = self.dsn4(self.side_up4(conv4))
        d5 = self.dsn5(self.side_up5(conv5))


        # dsn fusion output
        concat = torch.cat((d1, d2, d3, d4, d5), 1)
        bound = torch.cat([concat,x],1)
        bound = self.dconv_fuse_boundary(bound)
        
        mask = torch.cat([bound, x],1)
        mask = self.dconv_fuse_mask(mask)
        
        bound = self.final_boundary(bound)
        mask = self.final_mask(mask)
        
        sum_output = bound + mask
        binary =  sum_output > 0  
        return d1, d2, d3, d4, d5, bound, mask, sum_output, binary
    
