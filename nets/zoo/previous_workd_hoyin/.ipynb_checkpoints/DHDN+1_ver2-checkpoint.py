import os
import torch
from torch import nn
from torchvision.models import vgg16



###DHDN Layer 1층 추가

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
        
    """
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        return out
    """
    def forward(self, x):
        residual = x
        out = self.relu1(self.bn_1(self.conv_1(x)))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.bn_2(self.conv_2(conc)))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.bn_3(self.conv_3(conc)))
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
    def __init__(self, ch_para):
        super(_up, self).__init__()
#        self.relu = nn.PReLU()
#        self.subpixel = nn.PixelShuffle(2)
#        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)
        
        channel_in, channel_out = ch_para[0], ch_para[1]
        self.deconv = nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=2, padding=1, output_padding=1)
# 3x3 유지, stride, padding 조절
        self.bn_i = nn.BatchNorm2d(num_features=channel_out)
        self.relu = nn.ReLU()

    def forward(self, x):
#        out = self.relu(self.conv(x))
#        out = self.subpixel(out)
        
        out = self.bn_i(self.deconv(x))
        out = self.relu(out)
        
        return out

class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()

        self.conv_i = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, 32)
        self.DCR_block12 = self.make_layer(_DCR_block, 32)
        self.down1 = self.make_layer(_down, 32)
        self.DCR_block21 = self.make_layer(_DCR_block, 64)
        self.DCR_block22 = self.make_layer(_DCR_block, 64)
        self.down2 = self.make_layer(_down, 64)
        self.DCR_block31 = self.make_layer(_DCR_block, 128)
        self.DCR_block32 = self.make_layer(_DCR_block, 128)
        self.down3 = self.make_layer(_down_retain, 128)
        self.DCR_block41 = self.make_layer(_DCR_block, 128)
        self.DCR_block42 = self.make_layer(_DCR_block, 128)
        
        self.down4 = self.make_layer(_down_retain, 128)
        self.DCR_block51 = self.make_layer(_DCR_block, 128)
        self.DCR_block52 = self.make_layer(_DCR_block, 128)
        
        
        self.up4 = self.make_layer(_up, [256,128])
        
        self.DCR_block43 = self.make_layer(_DCR_block, 256)
        self.DCR_block44 = self.make_layer(_DCR_block, 256)
               
        self.up3 = self.make_layer(_up, [256, 128])
        
        self.DCR_block33 = self.make_layer(_DCR_block, 256)
        self.DCR_block34 = self.make_layer(_DCR_block, 256)
        self.up2 = self.make_layer(_up, [256,64])
        self.DCR_block23 = self.make_layer(_DCR_block, 128)
        self.DCR_block24 = self.make_layer(_DCR_block, 128)
        self.up1 = self.make_layer(_up, [128, 32])
        self.DCR_block13 = self.make_layer(_DCR_block, 64)
        self.DCR_block14 = self.make_layer(_DCR_block, 64)
        self.conv_f = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()
        #super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
 


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
        conc5 = self.down4(conc4)           #256 
        out = self.DCR_block51(conc5)       #256
        out = self.DCR_block52(out)         #256
        
        out = torch.cat([conc5, out], 1)    #256+256=512
        out = self.up4(out)                 #512->256
        out = torch.cat([conc4, out], 1)    #256+256=512
        out = self.DCR_block43(out)         #512
        out = self.DCR_block44(out)         #512
        
        out = self.up3(out)                 #512->256
        out = torch.cat([conc3, out], 1)    #256+256=512
        out = self.DCR_block33(out)         #512
        out = self.DCR_block34(out)         #512
        out = self.up2(out)                 #512 -> 128   
        out = torch.cat([conc2, out], 1)    #128+128=256
        out = self.DCR_block23(out)         #256
        out = self.DCR_block24(out)         #256
        out = self.up1(out)                 #256->64    
        out = torch.cat([conc1, out], 1)    #64+64=128
        out = self.DCR_block13(out)         #128
        out = self.DCR_block14(out)         #128
        out = self.relu2(self.conv_f(out))
        #out = torch.add(residual, out)

        return out


"""
ASPP(256, [12, 24, 36]),
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

"""
"""

class EvoNorm(nn.Module):
    def __init__(self, input, non_linear = True, version = 'S0', momentum = 0.9, eps = 1e-5, training = True):
        super(EvoNorm, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
        if self.non_linear:
            self.v = nn.Parameter(torch.ones(1,self.insize,1,1))
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
"""
#DHDN 원본 layer 추가 전
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
        out = torch.cat([conc4, out], 1)
        out = self.up3(out)
        out = torch.cat([conc3, out], 1)
        out = self.DCR_block33(out)
        out = self.DCR_block34(out)
        out = self.up2(out)
        out = torch.cat([conc2, out], 1)
        out = self.DCR_block23(out)
        out = self.DCR_block24(out)
        out = self.up1(out)
        out = torch.cat([conc1, out], 1)
        out = self.DCR_block13(out)
        out = self.DCR_block14(out)
        out = self.relu2(self.conv_f(out))
        #out = torch.add(residual, out)

        return out
"""
######################################################        SpaceNet Baseline




"""
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
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
        self.dec1 = XDXD_SN4_ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
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
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        return x_out


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

"""
       
        
"""
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.PReLU()
        self.DCR_block11 = self.make_layer(_DCR_block, 64)
        self.DCR_block12 = self.make_layer(_DCR_block, 64)
        self.down1 = self.make_layer(_down, 64)
        self.DCR_block21 = self.make_layer(_DCR_block, 128)
        self.DCR_block22 = self.make_layer(_DCR_block, 128)
        self.down2 = self.make_layer(_down, 128)
        self.DCR_block31 = self.make_layer(_DCR_block, 256)
        self.DCR_block32 = self.make_layer(_DCR_block, 256)
        self.down3 = self.make_layer(_down_retain, 256)
        self.DCR_block41 = self.make_layer(_DCR_block, 256)
        self.DCR_block42 = self.make_layer(_DCR_block, 256)
        
        self.down4 = self.make_layer(_down_retain, 256)
        self.DCR_block51 = self.make_layer(_DCR_block, 256)
        self.DCR_block52 = self.make_layer(_DCR_block, 256)
        
        
        self.up4 = self.make_layer(_up, [512,256])
        
        self.DCR_block43 = self.make_layer(_DCR_block, 512)
        self.DCR_block44 = self.make_layer(_DCR_block, 512)
               
        self.up3 = self.make_layer(_up, [512, 256])
        
        self.DCR_block33 = self.make_layer(_DCR_block, 512)
        self.DCR_block34 = self.make_layer(_DCR_block, 512)
        self.up2 = self.make_layer(_up, [512,128])
        self.DCR_block23 = self.make_layer(_DCR_block, 256)
        self.DCR_block24 = self.make_layer(_DCR_block, 256)
        self.up1 = self.make_layer(_up, [256, 64])
        self.DCR_block13 = self.make_layer(_DCR_block, 128)
        self.DCR_block14 = self.make_layer(_DCR_block, 128)
        self.conv_f = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.PReLU()
"""