import os
import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

#__all__ = ['Res2Net', 'res2net50']

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
            out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    #def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
    def __init__(self, block, layers, baseWidth = 26, scale = 4):        
        
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        con1 = self.relu(x)
        x = self.maxpool(con1)

        con2 = self.layer1(x)
        con3 = self.layer2(con2)
        con4 = self.layer3(con3)
        con5 = self.layer4(con4)

        return [con1, con2, con3, con4, con5, residual]    

    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
                
        self.model_res2_archi = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4)
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=num_filters*5, kernel_size=1, stride=1, padding=0)
        self.dec4 = DecoderBlock(num_filters*64, num_filters*32)
        self.dec3 = DecoderBlock(num_filters*64, num_filters*16)
        self.dec2 = DecoderBlock(num_filters*32, num_filters*8)
        self.dec1 = DecoderBlock(num_filters*16, num_filters*8)
        self.dec0 = DecoderBlock(num_filters*10, num_filters*5)
        self.conv_f = nn.Conv2d(in_channels=num_filters*10, out_channels=1, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        
        residual = x        
        temp_ret = self.model_res2_archi(x)
        enc1, enc2, enc3, enc4, enc5, residual = temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3], temp_ret[4], temp_ret[5]

        residual_conv = self.conv_i(residual)
        out4 = self.dec4(enc5)              
        conc4 = torch.cat([enc4, out4], 1)  
        out3 = self.dec3(conc4)             
        conc3 = torch.cat([enc3, out3], 1)  
        out2 = self.dec2(conc3)             
        conc2 = torch.cat([enc2, out2], 1)  
        out1 = self.dec1(conc2)             
        conc1 = torch.cat([enc1, out1], 1)  
        out0 = self.dec0(conc1)             
        conc0 = torch.cat([residual_conv, out0], 1)  
        out = self.conv_f(conc0)
        

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
    

class ResiRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv1 = nn.Conv2d(in_, out, 3, padding=1)
        self.activation1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_, out, 3, padding=1)
        self.activation2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_, out, 3, padding=1)
        self.activation3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        x = self.activation1( self.conv1(x) )
        x = self.activation2( self.conv2(x) )
        x = self.activation3( self.conv3(x) )
        
        return_val = x + residual
        return return_val
    


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvRelu(in_channels, out_channels),
            ResiRelu(out_channels, out_channels),
            ConvRelu(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

    
    
    
    
    