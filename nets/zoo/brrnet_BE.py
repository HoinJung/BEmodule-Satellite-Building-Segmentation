import torch
import torch.nn as nn
import skimage
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )   


def up_transpose(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)    
    )
class center_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(center_block, self).__init__()     
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2,dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=4,dilation=4)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=8,dilation=8)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=16,dilation=16)
        self.conv6 = nn.Conv2d(out_channels, out_channels, 3, padding=32,dilation=32)
        
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_2 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_4 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_5 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_6 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        
        
        
    def forward(self,x):# 지금 rrm쪽이랑 센터랑 섞임..
        
        
        x1 = self.relu(self.bn_1(self.conv1(x)))
        
        x2 = self.relu(self.bn_2(self.conv2(x1)))
        
        x3 = self.relu(self.bn_3(self.conv3(x2)))
        
        x4 = self.relu(self.bn_4(self.conv4(x3)))
        
        x5 = self.relu(self.bn_5(self.conv5(x4)))
        
        x6 = self.relu(self.bn_6(self.conv6(x5)))
        
        
        x = x1+x2+x3+x4+x5+x6
        
        return x
    
class rrm_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(rrm_module,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2,dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=4,dilation=4)
        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, padding=8,dilation=8)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 3, padding=16,dilation=16)
        self.conv6 = nn.Conv2d(out_channels, out_channels, 3, padding=32,dilation=32)
        
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_2 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_4 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_5 = nn.BatchNorm2d(num_features=out_channels)
        self.bn_6 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        
#         self.out = nn.Conv2d(out_channels, 1, 3, padding=1,dilation=1)
#         BE mode
        self.out = nn.Conv2d(out_channels, 64, 3, padding=1,dilation=1)
        
    def forward(self,x):
        residual = x
        x1 = self.relu(self.bn_1(self.conv1(x)))
        
        x2 = self.relu(self.bn_2(self.conv2(x1)))
        x3 = self.relu(self.bn_3(self.conv3(x2)))
        x4 = self.relu(self.bn_4(self.conv4(x3)))
        x5 = self.relu(self.bn_5(self.conv5(x4)))
        x6 = self.relu(self.bn_6(self.conv6(x5)))
        x = x1+x2+x3+x4+x5+x6
        x = self.out(x)
        x = residual + x
        output = x
#         output = F.sigmoid(x)
        return output        
    
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block,self).__init__()        
        self.bn_i = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.conv = conv_block(in_channels, out_channels)
    def forward(self, x):
        
        out = self.bn_i(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

class BRRNet_BE(nn.Module):

    def __init__(self, n_class=1, pretrained=False, mode= 'Train'):
        super().__init__()
        self.mode = mode
        self.dconv_down1 = conv_block(3, 64)
        self.dconv_down2 = conv_block(64, 128)
        self.dconv_down3 = conv_block(128, 256)

        self.maxpool = nn.MaxPool2d(2,2)
        self.center = center_block(256,512)
        self.deconv3 = up_transpose(512,256)
        self.deconv2 = up_transpose(256,128)
        self.deconv1 = up_transpose(128,64)
        
        self.decoder_3 = decoder_block(512, 256)
        self.decoder_2 = decoder_block(256, 128)
        self.decoder_1 = decoder_block(128, 64)
#         self.output_1 = nn.Conv2d(64,n_class, 1)
#         self.rrm = rrm_module(1,64)
        # BE mode
        self.output_1 = nn.Conv2d(64,64, 1)
        self.rrm = rrm_module(64,64)
        
        # HED Block
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        

        #boundary enhancement part        
        self.fuse = nn.Sequential(nn.Conv2d(4, 64, 1),nn.ReLU(inplace=True))

        self.SE_mimic = nn.Sequential(        
            nn.Linear(64, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4, bias=False),
            nn.Sigmoid()
        )
        self.final_boundary = nn.Conv2d(4,2,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(128,64,3, padding=1),
            nn.ReLU(inplace=True)            
        )
        self.final_mask = nn.Conv2d(64,2,1)
        
                

        self.relu = nn.ReLU()        
        self.out = nn.Conv2d(64,1,1)
        

        
        
        
    def forward(self, x):
        h = x.size(2)
        w = x.size(3)
        conv1 = self.dconv_down1(x)
#         print(conv1.shape)
        x = self.maxpool(conv1)
#         print(x.shape)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.center(x)
        
        x = self.deconv3(conv4) # 512 256
        x = torch.cat([conv3,x],1) # 256 + 256
        
        x = self.decoder_3(x) # 512 256
        
        x = self.deconv2(x)
        x = torch.cat([conv2,x],1)
        x = self.decoder_2(x)
        
        x = self.deconv1(x)
        x = torch.cat([conv1,x],1)
        x = self.decoder_1(x)
        
        x = self.output_1(x)
        out = self.rrm(x)
        
        
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
        
        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        
        concat = torch.cat((d1_out, d2_out, d3_out, d4_out), 1)
        
        fuse_box = self.fuse(concat)
        GAP = F.adaptive_avg_pool2d(fuse_box,(1,1))
        GAP = GAP.view(-1, 64)        
        se_like = self.SE_mimic(GAP)        
        se_like = torch.unsqueeze(se_like, 2)
        se_like = torch.unsqueeze(se_like, 3)

        feat_se = concat * se_like.expand_as(concat)
        boundary = self.final_boundary(feat_se)
        boundary_out = torch.unsqueeze(boundary[:,1,:,:],1)        
        bd_sftmax = F.softmax(boundary, dim=1)
        boundary_scale = torch.unsqueeze(bd_sftmax[:,1,:,:],1)        
        
        feat_concat = torch.cat( [out, fuse_box], 1)
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

            return d1_out, d2_out, d3_out, d4_out, boundary_out, mask_out
        elif self.mode =='Infer':
            return mask_out
        
        