import torch
import torch.nn as nn
import skimage
import numpy as np
from torch.autograd import Variable

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

class UNet_assemble(nn.Module):

    def __init__(self, n_class=1, pretrained=False):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.dconv_down5 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.upsample5 = _up_deconv(1024,512)        
        self.upsample4 = _up_deconv(512,256)
        self.upsample3 = _up_deconv(256,128)
        self.upsample2 = _up_deconv(128,64)
        self.dconv_up4 = double_conv(512 + 512, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.conv3_b = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_b2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_m = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_b = nn.Conv2d(64, 1, 1)
        self.conv1_m = nn.Conv2d(64, 1, 1)
        
        self.boundary_test = double_conv(64, 64)
        
        
        self.relu = nn.ReLU()
#         self.boundary = boundary_extraction()
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        x = self.dconv_down5(x)        
        x = self.upsample5(x)        
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
        
        
        x = self.boundary_test(x)
        out_b = self.conv1_b(x)
        out_b = boundary_extraction(out_b)
        out_m = self.conv3_m(x)
        out_m = self.conv1_m(out_m)
        return out_m, out_b
    
    
    
def boundary_extraction(mask):
    
    # obtain boundray from 1-channel mask
    arr_mask = mask.cpu().detach().numpy()
    binary = arr_mask > 0 
    mask_boundary_arr =  skimage.segmentation.find_boundaries(binary, mode='inner', background=0).astype(np.float32) 
    mask_boundary = torch.from_numpy(mask_boundary_arr).cuda().float()
    mask_boundary = Variable(mask_boundary, requires_grad=True)

    return mask_boundary
            