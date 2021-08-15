import torch
import torch.nn as nn
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
        
        self.out = nn.Conv2d(out_channels, 1, 3, padding=1,dilation=1)
        
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
        
        return x        
    
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

class BRRNet(nn.Module):

    def __init__(self, n_class=1, pretrained=False,mode='Train'):
        super().__init__()
        self.mode=mode        
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
        self.output_1 = nn.Conv2d(64,n_class, 1)
        self.rrm = rrm_module(1,64)
    def forward(self, x):
        
        conv1 = self.dconv_down1(x)
#         print(conv1.shape)
        x = self.maxpool(conv1)
#         print(x.shape)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.center(x)
        
        x = self.deconv3(x) # 512 256
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
        if self.mode == 'Train':
            return F.sigmoid(out)
        elif self.mode == 'Infer':
            return out