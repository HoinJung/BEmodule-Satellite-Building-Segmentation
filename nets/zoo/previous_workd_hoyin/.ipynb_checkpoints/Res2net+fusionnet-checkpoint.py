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
        out = EvoNorm(out)
        #out = self.bn1(out)
        #out = self.relu(out)

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
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        
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
        x = self.conv1(x)
        con1=EvoNorm(x)
        #x = self.bn1(x)
        #con1 = self.relu(x)
        x = self.maxpool(con1)

        con2 = self.layer1(x)
        con3 = self.layer2(con2)
        con4 = self.layer3(con3)
        con5 = self.layer4(con4)

        return [con1, con2, con3, con4, con5]
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        #return x
    

    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
                
        self.model_res2_archi = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4)
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(512, 128)
        self.dec2 = DecoderBlock(256, 64)
        self.dec1 = DecoderBlock(128, 64)
        
        self.conv_f = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()
  
        self.ASPP=self.make_layer(ASPP, 512)
    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    
    def forward(self, x):
        
        residual = x
        
        temp_ret = self.model_res2_archi(x)
        enc1, enc2, enc3, enc4, enc5 = temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3], temp_ret[4]
        
        out4 = self.dec4(enc5)              #512=>256
        conc4 = torch.cat([enc4, out4], 1)  #256+256
        out3 = self.dec3(conc4)             #512=>128
        conc3 = torch.cat([enc3, out3], 1)  #128+128
        out2 = self.dec2(conc3)             #256=>64
        conc2 = torch.cat([enc2, out2], 1)  #64+64
        out1 = self.dec1(conc2)             #128=>64
        conc1 = torch.cat([enc1, out1], 1)  #64+64
        self.conv_f(conc1)
                

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

    
    
    
    
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels):    
        super(ASPP, self).__init__()
        out_channels = 512
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
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            EvoNorm(out_channels))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def group_std(x, groups = 32, eps = 1e-5):
#def group_std(x, groups = 16, eps = 1e-5):
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
            self.gamma = nn.Parameter(torch.ones(1, input, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input, 1, 1))
            if self.non_linear:
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
