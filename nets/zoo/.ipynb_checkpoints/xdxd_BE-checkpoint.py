import os
import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
import numpy as np

class XDXD_BE(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super().__init__()
        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        self.center = XDXD_SN4_DecoderBlock(512, num_filters * 8 * 2,
                                            num_filters * 8)
        self.dec5 = XDXD_SN4_DecoderBlock(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = XDXD_SN4_DecoderBlock(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = XDXD_SN4_DecoderBlock(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = XDXD_SN4_DecoderBlock(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = XDXD_SN4_ConvRelu(64 + num_filters, num_filters)#96 32 3x3 -> ReLU -> 1x1 32 1
#         self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

        
                # HED Block
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(5, 32, 1),nn.ReLU(inplace=True))
#         self.fuse = nn.Conv2d(5, 64, 1)
        
        self.SE_mimic = nn.Sequential(        
            nn.Linear(32, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5, bias=False),
            nn.Sigmoid()
        )
        self.final_boundary = nn.Conv2d(5,2,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True)            
        )
        self.final_mask = nn.Conv2d(64,2,1)
        
                

        self.relu = nn.ReLU()        
        self.out = nn.Conv2d(64,1,1)
        
#         self.mode = 'Train'
        self.mode = 'Infer'

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        x = self.dec1(torch.cat([dec2, conv1], 1))
#         dec1 = self.dec1(torch.cat([dec2, conv1], 1))
#         x_out = self.final(dec1)
#         x_out_2 = x_out
        
        
         ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))
#
        ###########sigmoid ver
        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        d5_out = F.sigmoid(d5)

        concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out), 1)

        fuse_box = self.fuse(concat)
        GAP = F.adaptive_avg_pool2d(fuse_box,(1,1))
        GAP = GAP.view(-1, 32)        
        se_like = self.SE_mimic(GAP)        
        se_like = torch.unsqueeze(se_like, 2)
        se_like = torch.unsqueeze(se_like, 3)

        feat_se = concat * se_like.expand_as(concat)
        boundary = self.final_boundary(feat_se)
        boundary_out = torch.unsqueeze(boundary[:,1,:,:],1)        
        bd_sftmax = F.softmax(boundary, dim=1)
        boundary_scale = torch.unsqueeze(bd_sftmax[:,1,:,:],1)        
        
        feat_concat = torch.cat( [x, fuse_box], 1)
        feat_concat_conv = self.final_conv(feat_concat)
        mask = self.final_mask(feat_concat_conv)
        mask_sftmax = F.softmax(mask,dim=1)        
        mask_scale = torch.unsqueeze(mask_sftmax[:,1,:,:],1)

        if self.mode == 'Train':
            scalefactor = torch.clamp(mask_scale+boundary_scale,0,1)            
        elif self.mode == 'Infer':
            scalefactor = torch.clamp(mask_scale+5*boundary_scale,0,1)
        
        
        mask_out = torch.unsqueeze(mask[:,1,:,:],1)
        relu = self.relu(mask_out)        
        scalar = relu.cpu().detach().numpy()
        if np.sum(scalar) == 0:
            average = 0
        else : 
            average = scalar[np.nonzero(scalar)].mean()
        mask_out = mask_out-relu + (average*scalefactor)
        
        if self.mode == 'Train':
            mask_out = F.sigmoid(mask_out)
            boundary_out = F.sigmoid(boundary_out)
            
            return d1_out, d2_out, d3_out, d4_out, d5_out, boundary_out, mask_out
        elif self.mode =='Infer':
            return d1_out, d2_out, d3_out, d4_out, d5_out,boundary_scale, mask_out, mask_scale, scalefactor

        
        
class XDXD_SN4_ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class XDXD_SN4_DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(XDXD_SN4_DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            XDXD_SN4_ConvRelu(in_channels, middle_channels),
            XDXD_SN4_ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

# below dictionary lists models compatible with solaris. alternatively, your
# own model can be used by using the path to the model as the value for
# model_name in the config file.
