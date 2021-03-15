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
    
class BE_unet(nn.Module):

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
        
        # HED Block
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(1024, 1, 1)
        
        #boundary enhancement part        
        self.fuse = nn.Conv2d(5, 64, 3, padding=1)   
        self.fc_gap = nn.Linear(64, 5)
        self.final_boundary = nn.Conv2d(5,1,1)
        
        self.relu = nn.ReLU()        
        self.dconv_fuse_mask = double_conv(64,64)
        self.final_mask = nn.Conv2d(64,1,1)

        
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
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))

        # dsn fusion output
        concat = torch.cat((d1, d2, d3, d4, d5), 1)
        fuse_box = self.fuse(concat)
        print(fuse_box.shape)
        GAP = F.avg_pool2d(fuse_box,512)
        print(GAP.shape)
        GAP=GAP.view(fuse_box.size(0),-1)
        print(GAP.shape)
        
#         print(GAP.shape)
        fc_gap = F.relu(self.fc_gap(GAP))
        boundary = torch.mul(fc_gap,concat)
#         print(fc_gap)
#         print(fc_gap.shape)
#         print(fc_gap[0,0])
#         print(fc_gap[1,0])
#         d1 = d1 * fc_gap[0,0].item()
#         d2 = d2 * fc_gap[x,1].item()
#         d3 = d3 * fc_gap[2].item()
#         d4 = d4 * fc_gap[3].item()
#         d5 = d5 * fc_gap[4].item()



#         boundary = torch.cat((d1,d2,d3,d4,d5),1)
        bound = self.final_boundary(boundary)
        mask = x + fuse_box
        mask = self.dconv_fuse_mask(mask)
        mask = self.final_mask(mask)
        
        sum_output = bound + mask
        binary =  mask > 0  
        return d1, d2, d3, d4, d5, bound, mask, sum_output, binary
    
