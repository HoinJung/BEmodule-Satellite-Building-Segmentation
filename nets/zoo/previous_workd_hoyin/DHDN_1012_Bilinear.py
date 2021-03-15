import os
import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
import torch.nn as nn

class _DCR_block(nn.Module):
    def __init__(self, channel_in):        
        super(_DCR_block, self).__init__()       
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=channel_in)
        self.relu3 = nn.PReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        return out

class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.relu(self.conv(out))
        return out
        
class _down_retain(nn.Module):
    def __init__(self, channel_in):
        super(_down_retain, self).__init__()
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)
        out = self.relu(self.conv(out))
        return out

class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()
        self.relu = nn.PReLU()
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
    

class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()

        self.conv_i = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, 16)
        self.DCR_block12 = self.make_layer(_DCR_block, 16)
        self.down1 = self.make_layer(_down, 16)
        self.DCR_block21 = self.make_layer(_DCR_block, 32)
        self.DCR_block22 = self.make_layer(_DCR_block, 32)
        self.down2 = self.make_layer(_down, 32)
        self.DCR_block31 = self.make_layer(_DCR_block, 64)
        self.DCR_block32 = self.make_layer(_DCR_block, 64)
        self.down3 = self.make_layer(_down, 64)
        self.DCR_block41 = self.make_layer(_DCR_block, 128)
        self.DCR_block42 = self.make_layer(_DCR_block, 128)
        self.down4 = self.make_layer(_down_retain, 128)
        self.DCR_block51 = self.make_layer(_DCR_block, 128)
        self.DCR_block52 = self.make_layer(_DCR_block, 128)
        #self.ASPP=self.make_layer(ASPP, 256)
        self.up4 = self.make_layer(_up, 256)
        #self.up_deconv4=self.make_layer(_up_deconv,[256,64])
        #self.up_deconv4=DecoderBlock(256,160,64)
        self.up_deconv4=self.make_layer(DecoderBlock,[256,128,64])
        self.DCR_block43 = self.make_layer(_DCR_block, 256)
        self.DCR_block44 = self.make_layer(_DCR_block, 256)
        self.up3 = self.make_layer(_up, 256)
        #self.up_deconv3=self.make_layer(_up_deconv,[256,64])
        #self.up_deconv3=DecoderBlock(256,160,64)
        self.up_deconv3=self.make_layer(DecoderBlock,[256,128,64])
        self.DCR_block33 = self.make_layer(_DCR_block, 192)
        self.DCR_block34 = self.make_layer(_DCR_block, 192)
        self.up2 = self.make_layer(_up, 192)
        #self.up_deconv2=self.make_layer(_up_deconv,[192,48])
        #self.up_deconv2=DecoderBlock(192,120,48)
        self.up_deconv2=self.make_layer(DecoderBlock,[192,96,48])
        self.DCR_block23 = self.make_layer(_DCR_block, 128)
        self.DCR_block24 = self.make_layer(_DCR_block, 128)
        self.up1 = self.make_layer(_up, 128)
        #self.up_deconv1=self.make_layer(_up_deconv,[128, 32])
        #self.up_deconv1=DecoderBlock(128,80,32)
        self.up_deconv1=self.make_layer(DecoderBlock,[128,64,32])
        self.DCR_block13 = self.make_layer(_DCR_block, 80)
        self.DCR_block14 = self.make_layer(_DCR_block, 80)
        self.conv_f1 = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.conv_f2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
      
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
        out = self.down3(conc3)
        out = self.DCR_block41(out)
        conc4 = self.DCR_block42(out)
        conc5 = self.down4(conc4)           
        out = self.DCR_block51(conc5)       
        out = self.DCR_block52(out)         
        
        out = torch.cat([conc5, out], 1)    
        #out = self.ASPP(out)
        
        out_s_4 = self.up4(out)                 
        out_d_4 = self.up_deconv4(out)          
        out = torch.cat([conc4, out_d_4], 1)    
        out = torch.cat([out, out_s_4], 1)
        out = self.DCR_block43(out)         
        out = self.DCR_block44(out)       
        
        out_s_3 = self.up3(out)             
        out_d_3 = self.up_deconv3(out)              
        out = torch.cat([conc3, out_d_3], 1)
        out = torch.cat([out, out_s_3], 1)
        out = self.DCR_block33(out)         
        out = self.DCR_block34(out)         
        
        out_s_2 = self.up2(out)             
        out_d_2 = self.up_deconv2(out)    
        out = torch.cat([conc2, out_d_2], 1)    
        out = torch.cat([out, out_s_2], 1)    
        out = self.DCR_block23(out)        
        out = self.DCR_block24(out)
        
        out_s_1 = self.up1(out)            
        out_d_1 = self.up_deconv1(out)     
        out = torch.cat([conc1, out_d_1], 1)    
        out = torch.cat([out, out_s_1], 1)
        out = self.DCR_block13(out)         
        out = self.DCR_block14(out)         
        out = self.conv_f1(out)
        out = self.activation(out)
        out = self.conv_f2(out)

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

class DecoderBlock(nn.Module):
    def __init__(self, ch_para):
        super(DecoderBlock, self).__init__()        
        in_channels, middle_channels, out_channels = ch_para[0], ch_para[1], ch_para[2]
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)
"""    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)
"""
class ASPP(nn.Module):
    def __init__(self, in_channels):    
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            EvoNorm(out_channels)))
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU()))

        atrous_rates=[12, 24, 36]
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            EvoNorm(out_channels),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            EvoNorm(out_channels)
            #nn.BatchNorm2d(out_channels),        
            #nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            EvoNorm(out_channels))
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

"""    
#def group_std(x, groups = 32, eps = 1e-5):
def group_std(x, groups = 16, eps = 1e-5):
    N, C, H, W = x.size()
    
    temp_in = int(C // float(groups))
    
    x = torch.reshape(x, (N, groups, temp_in, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))

class EvoNorm(nn.Module):
    def __init__(self, input, non_linear = True, version = 'S0', momentum = 0.9, eps = 1e-5, training = True, affine=True):
        super(EvoNorm, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input

        self.affine = affine
        
        if self.affine:
            #self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            #self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            self.gamma = nn.Parameter(torch.ones(1, input, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input, 1, 1))
            if self.non_linear:
                #self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
                self.v = nn.Parameter(torch.ones(1,input,1,1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        
        self.register_buffer('running_var', torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)
    
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))
        if self.version == 'S0':
            if self.non_linear:
                num = x * torch.sigmoid(self.v * x)
                return num / group_std(x, eps = self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = torch.var(x, dim = (0, 2, 3), unbiased = False, keepdim = True).reshape(1, x.size(1), 1, 1)
                with torch.no_grad():
                    self.running_var.copy_(self.momentum * self.running_var + (1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std(x, eps = self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
"""