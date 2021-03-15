import os
import torch
from torch import nn
from torchvision.models import vgg16





class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=channel_in)
        self.relu3 = nn.ReLU()
        

    def forward(self, x):
        residual = x
        out = self.bn_1(self.conv_1(x))
        out = self.relu1(out)
        conc = torch.cat([x, out], 1)
        out = self.bn_2(self.conv_2(conc))
        out = self.relu2(out)
        conc = torch.cat([conc, out], 1)
        out = self.bn_3(self.conv_3(conc))
        out = self.relu3(out)
        out = torch.add(out, residual)
        return out

class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.relu(self.conv(out))
        return out
        

class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()
        self.relu = nn.ReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.subpixel(out)
        return out

    
class _up_deconv(nn.Module):
    def __init__(self, ch_para):
        super(_up_deconv, self).__init__()        
        channel_in, channel_out = ch_para[0], ch_para[1]
        self.deconv = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_i = nn.BatchNorm2d(num_features=channel_out)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.bn_i(self.deconv(x))
        out = self.relu(out)
        
        return out
    
    
class _DHDN(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(_DHDN, self).__init__()

        self.conv_i = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, num_filters)
        self.DCR_block12 = self.make_layer(_DCR_block, num_filters)
        self.down1 = self.make_layer(_down, num_filters)
        self.DCR_block21 = self.make_layer(_DCR_block, num_filters*2)
        self.DCR_block22 = self.make_layer(_DCR_block, num_filters*2)
        self.down2 = self.make_layer(_down, num_filters*2)
        self.DCR_block31 = self.make_layer(_DCR_block, num_filters*4)
        self.DCR_block32 = self.make_layer(_DCR_block, num_filters*4)
        self.down3 = self.make_layer(_down, num_filters*4)
        self.DCR_block41 = self.make_layer(_DCR_block, num_filters*8)
        self.DCR_block42 = self.make_layer(_DCR_block, num_filters*8)
        self.up3 = self.make_layer(_up, num_filters*16)
        self.up_deconv3=self.make_layer(_up_deconv,[num_filters*16,num_filters*4])
        self.DCR_block33 = self.make_layer(_DCR_block, num_filters*12)
        self.DCR_block34 = self.make_layer(_DCR_block, num_filters*12)        
        self.up2 = self.make_layer(_up, num_filters*12)
        self.up_deconv2=self.make_layer(_up_deconv,[num_filters*12,num_filters*3])                
        self.DCR_block23 = self.make_layer(_DCR_block, num_filters*8)
        self.DCR_block24 = self.make_layer(_DCR_block, num_filters*8)        
        self.up1 = self.make_layer(_up, num_filters*8)
        self.up_deconv1=self.make_layer(_up_deconv,[num_filters*8,num_filters*2])    
        
        self.DCR_block13 = self.make_layer(_DCR_block, num_filters*5)
        self.DCR_block14 = self.make_layer(_DCR_block, num_filters*5)
        
        self.conv_f = nn.Conv2d(in_channels=num_filters*5, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_i(x))        
        out = self.DCR_block11(out)
        conc1 = self.DCR_block12(out)
        out = self.down1(conc1)
        out = self.DCR_block21(out)
        conc2 = self.DCR_block22(out)
        out = self.down2(conc2)
        out = self.DCR_block31(out)
        conc3 = self.DCR_block32(out)
        conc4 = self.down3(conc3)
        out = self.DCR_block41(conc4)
        out = self.DCR_block42(out)
        out4 = torch.cat([conc4, out], 1)   
        out_s_3 = self.up3(out4)             
        out_d_3 = self.up_deconv3(out4)      
        out = torch.cat([conc3, out_d_3], 1)
        out = torch.cat([out, out_s_3], 1)
        out = self.DCR_block33(out)        
        out3 = self.DCR_block34(out)        
        out_s_2 = self.up2(out3)                   
        out_d_2 = self.up_deconv2(out3)
        out = torch.cat([conc2, out_d_2], 1)
        out = torch.cat([out, out_s_2], 1)
        out = self.DCR_block23(out)         
        out2 = self.DCR_block24(out)
        out_s_1 = self.up1(out2)
        out_d_1 = self.up_deconv1(out2)    
        out = torch.cat([conc1, out_d_1], 1)
        out = torch.cat([out, out_s_1], 1)
        out = self.DCR_block13(out)         
        out = self.DCR_block14(out)         
        out = self.relu2(self.conv_f(out))
        

        return out

	    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
        
        self.encoder_upper = _DHDN()
        self.encoder_lower = _DHDN()
          
    
    def forward(self,x):
        upper_value = self.encoder_upper(x)
        lower_value = self.encoder_lower(x)
        
        return_val = upper_value - lower_value
        
        return return_val
