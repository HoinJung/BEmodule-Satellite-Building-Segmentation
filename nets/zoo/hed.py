# import libraries
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.models import vgg16

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class HED(nn.Module):
    def __init__(self, pretrained=False):
        super(HED, self).__init__()
#         self.encoder = vgg16(pretrained=pretrained).features
#         self.pool = nn.MaxPool2d(2, 2)

#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Sequential(
#             self.encoder[0], self.relu, self.encoder[2], self.relu)
#         self.conv2 = nn.Sequential(
#             self.encoder[5], self.relu, self.encoder[7], self.relu)
#         self.conv3 = nn.Sequential(
#             self.encoder[10], self.relu, self.encoder[12], self.relu,
#             self.encoder[14], self.relu)
#         self.conv4 = nn.Sequential(
#             self.encoder[17], self.relu, self.encoder[19], self.relu,
#             self.encoder[21], self.relu)
#         self.conv5 = nn.Sequential(
#             self.encoder[24], self.relu, self.encoder[26], self.relu,
#             self.encoder[28], self.relu)
#         # conv1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # conv2
#         self.conv2 = nn.Sequential(
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # conv3
#         self.conv3 = nn.Sequential(
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # conv4
#         self.conv4 = nn.Sequential(
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # conv5
#         self.conv5 = nn.Sequential(
#             nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.dconv_down5 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)                
        
        # HED Block
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(1024, 1, 1)
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

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
       

        ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))

        # dsn fusion output
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))
        
#         d1 = F.sigmoid(d1)
#         d2 = F.sigmoid(d2)
#         d3 = F.sigmoid(d3)
#         d4 = F.sigmoid(d4)
#         d5 = F.sigmoid(d5)
#         fuse = F.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse
