import torch
import torch.nn as nn
from torch.nn import functional as F


class _stage_block(nn.Module):
    def __init__(self, channel_var):
        super(_stage_block, self).__init__()
        
        channel_in, channel_out = channel_var
        
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU()        
        
    def forward(self, x):
        out = self.bn( self.conv(x) )
        out = self.relu(out)        
        return out    


class _upss_block(nn.Module):
    def __init__(self, channel_in):
        super(_upss_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.MaxPool2d(1),
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=1, stride=1, padding=0),
        )        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=2, stride=1, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=3, stride=1, padding=1),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(6),
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/4.), kernel_size=4, stride=1, padding=2),
        )

    def forward(self, x):
        residual = x
        
        h, w = x.size(2), x.size(3)
        
        out1 = self.conv1(x)        
        out1 = F.upsample(input=out1, size=(h, w), mode='bilinear')
        out2 = self.conv2(x)        
        out2 = F.upsample(input=out2, size=(h, w), mode='bilinear')
        out3 = self.conv3(x)        
        out3 = F.upsample(input=out3, size=(h, w), mode='bilinear')
        out4 = self.conv4(x)        
        out4 = F.upsample(input=out4, size=(h, w), mode='bilinear')
        
        out = torch.cat([out1, out2, out3, out4, residual], 1)
        return out
    
    
class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)       

    def forward(self, x):
        out = self.maxpool(x)
        return out


class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        #self.relu = nn.PReLU()
        #self.subpixel = nn.PixelShuffle(2)
        self.subpixel = nn.ConvTranspose2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=2, stride=2)
        #self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        #out = self.relu(self.conv(x))
        #out = self.subpixel(out)
        out = self.subpixel(x)
        return out

    
class Uspp(nn.Module):
    def __init__(self, pretrained=False,mode='Train'):
        super(Uspp, self).__init__()
        self.mode=mode
        self.DCR_block11 = self.make_layer(_stage_block, [  3, 64])
        self.DCR_block12 = self.make_layer(_stage_block, [ 64, 64])
        self.down1 = self.make_layer(_down, 64)
        self.DCR_block21 = self.make_layer(_stage_block, [ 64,128])
        self.DCR_block22 = self.make_layer(_stage_block, [128,128])
        self.down2 = self.make_layer(_down, 128)
        self.DCR_block31 = self.make_layer(_stage_block, [128,256])
        self.DCR_block32 = self.make_layer(_stage_block, [256,256])
        self.down3 = self.make_layer(_down, 256)
        self.DCR_block41 = self.make_layer(_stage_block, [256,512])
        self.DCR_block42 = self.make_layer(_stage_block, [512,512])
        self.down4 = self.make_layer(_down, 512)
        
        self.uspp = self.make_layer(_upss_block, 512)
        
        self.up4 = self.make_layer(_up, 1024)
        self.DCR_block43 = self.make_layer(_stage_block,[1024,512])
        self.DCR_block44 = self.make_layer(_stage_block, [512,512])
        self.up3 = self.make_layer(_up, 512)
        self.DCR_block33 = self.make_layer(_stage_block, [512,256])
        self.DCR_block34 = self.make_layer(_stage_block, [256,256])
        self.up2 = self.make_layer(_up, 256)
        self.DCR_block23 = self.make_layer(_stage_block, [256,128])
        self.DCR_block24 = self.make_layer(_stage_block, [128,128])
        self.up1 = self.make_layer(_up, 128)
        self.DCR_block13 = self.make_layer(_stage_block, [128, 64])
        self.DCR_block14 = self.make_layer(_stage_block, [ 64,  1])

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        out = self.DCR_block11(x)
        conc1= self.DCR_block12(out)
        out = self.down1(conc1)
        
        out = self.DCR_block21(out)
        conc2= self.DCR_block22(out)
        out = self.down2(conc2)

        out = self.DCR_block31(out)
        conc3= self.DCR_block32(out)
        out = self.down3(conc3)

        out = self.DCR_block41(out)
        conc4= self.DCR_block42(out)
        out = self.down4(conc4)
        
        # bridge part
        out = self.uspp(out)

        out = self.up4(out)
        out = torch.cat([conc4, out], 1)
        out = self.DCR_block43(out)
        out = self.DCR_block44(out)
        
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
        if self.mode == 'Train':
            return F.sigmoid(out)
        elif self.mode == 'Infer':
            return out