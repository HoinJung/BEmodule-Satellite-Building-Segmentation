import torch
import torch.nn as nn
import torch.nn.functional as F
# working for double-U Net architecture =========================================

class _SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    
class _conv_only_block(nn.Module):
    def __init__(self, channel):
        super(_conv_only_block, self).__init__()
        
        channel_in = channel[0]
        channel_out = channel[1]
    
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out) 
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.bn1( self.conv(x) )
        
        return out
    
class _conv_block(nn.Module):
    def __init__(self, channel):
        super(_conv_block, self).__init__()
        
        channel_in = channel[0]
        channel_out = channel[1]
    
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out) 
        self.relu1 = nn.ReLU()
        
        self.se_block = self.make_layer(_SELayer, channel_out)
    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        residual = x
        
        # front convolution part
        out = self.bn1( self.conv(x) )
        out = self.relu1(out)
        
        out = self.se_block(out)
        
        return out
    
# class ASPP(nn.Module):
#     def __init__(self, channel_in, channel_out):
#         super(ASPP, self).__init__()

#         self.conv_1x1_1 = nn.Conv2d(channel_in, channel_in//2, kernel_size=1)
#         self.bn_conv_1x1_1 = nn.BatchNorm2d(channel_in//2)

#         self.conv_3x3_1 = nn.Conv2d(channel_in, channel_in//2, kernel_size=3, stride=1, padding=6, dilation=6)
#         self.bn_conv_3x3_1 = nn.BatchNorm2d(channel_in//2)

#         self.conv_3x3_2 = nn.Conv2d(channel_in, channel_in//2, kernel_size=3, stride=1, padding=12, dilation=12)
#         self.bn_conv_3x3_2 = nn.BatchNorm2d(channel_in//2)

#         self.conv_3x3_3 = nn.Conv2d(channel_in, channel_in//2, kernel_size=3, stride=1, padding=18, dilation=18)
#         self.bn_conv_3x3_3 = nn.BatchNorm2d(channel_in//2)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.conv_1x1_2 = nn.Conv2d(channel_in, channel_in//2, kernel_size=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(channel_in//2)

#         self.conv_1x1_3 = nn.Conv2d(channel_in*5//2, channel_in//2, kernel_size=1) # (1280 = 5*256)
#         self.bn_conv_1x1_3 = nn.BatchNorm2d(channel_in//2)

#         self.conv_1x1_4 = nn.Conv2d(channel_in//2, channel_out, kernel_size=1)

#     def forward(self, feature_map):
#         # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

#         feature_map_h = feature_map.size()[2] # (== h/16)
#         feature_map_w = feature_map.size()[3] # (== w/16)

#         out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
#         out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

#         out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
#         out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
#         out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

#         out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
#         out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
#         out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

#         return out
    
    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):    
        super(ASPP, self).__init__()
        #out_channels = 512
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        atrous_rates=[12, 24, 36]
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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        
    
class _vgg19(nn.Module):
    def __init__(self, temp_in):
        super(_vgg19, self).__init__()
    
        channel_in = 3
        channel_stage1 = 32
        channel_stage2 = 64
        channel_stage3 = 128
        channel_stage4 = 256
        
        # stage 1 part
        self.stage_1_1 = self.make_layer( _conv_block, [channel_in, channel_stage1] )
        self.stage_1_2 = self.make_layer( _conv_block, [channel_stage1, channel_stage1] )
        
        # stage 2 part
        self.stage_2_1 = self.make_layer( _conv_block, [channel_stage1, channel_stage2] )
        self.stage_2_2 = self.make_layer( _conv_block, [channel_stage2, channel_stage2] )
        
        # stage 3 part
        self.stage_3_1 = self.make_layer( _conv_block, [channel_stage2, channel_stage3] )
        self.stage_3_2 = self.make_layer( _conv_block, [channel_stage3, channel_stage3] )
        self.stage_3_3 = self.make_layer( _conv_block, [channel_stage3, channel_stage3] )
        self.stage_3_4 = self.make_layer( _conv_block, [channel_stage3, channel_stage3] )
        
        # stage 4 part
        self.stage_4_1 = self.make_layer( _conv_block, [channel_stage3, channel_stage4] )
        self.stage_4_2 = self.make_layer( _conv_block, [channel_stage4, channel_stage4] )
        self.stage_4_3 = self.make_layer( _conv_block, [channel_stage4, channel_stage4] )
        self.stage_4_4 = self.make_layer( _conv_block, [channel_stage4, channel_stage4] )
        
        self.maxpool = nn.MaxPool2d(2)
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #residual = x
        out = self.stage_1_1(x)
        out_st1 = self.stage_1_2(out)
        
        out = self.maxpool(out_st1)
        out = self.stage_2_1(out)
        out_st2 = self.stage_2_2(out)
        
        out = self.maxpool(out_st2)
        out = self.stage_3_1(out)
        out = self.stage_3_2(out)
        out = self.stage_3_3(out)
        out_st3 = self.stage_3_4(out)
        
        out = self.maxpool(out_st3)
        out = self.stage_4_1(out)
        out = self.stage_4_2(out)
        out = self.stage_4_3(out)
        out_st4 = self.stage_4_4(out)
        
        out = self.maxpool(out_st4)
        
        return out, [out_st1, out_st2, out_st3, out_st4]
    
class _dec_block(nn.Module):
    def __init__(self, dec_dep_set):
        super(_dec_block, self).__init__()
        
        dep1, dep2, dep3, dep4 = dec_dep_set[0], dec_dep_set[1], dec_dep_set[2], dec_dep_set[3]
        
        self.stage_4_1 = self.make_layer(_conv_block, [dep4, 256])
        self.stage_3_1 = self.make_layer(_conv_block, [dep3, 128])
        self.stage_2_1 = self.make_layer(_conv_block, [dep2, 64])
        
        self.stage_mid = self.make_layer(_conv_only_block, [dep1, 3])
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x_aspp, x_enc = x[0], x[1]
        out_st1, out_st2, out_st3, out_st4 = x_enc[0], x_enc[1], x_enc[2], x_enc[3]
        
        decon_41 = F.interpolate(x_aspp, size=out_st4.size()[2:], mode='bilinear', align_corners=True)
        concat_4 = torch.cat([decon_41, out_st4], 1)
        deconv_4 = self.stage_4_1(concat_4)
        
        decon_31 = F.interpolate(deconv_4, size=out_st3.size()[2:], mode='bilinear', align_corners=True)
        concat_3 = torch.cat([decon_31, out_st3], 1)
        deconv_3 = self.stage_3_1(concat_3)
        
        decon_21 = F.interpolate(deconv_3, size=out_st2.size()[2:], mode='bilinear', align_corners=True)
        concat_2 = torch.cat([decon_21, out_st2], 1)
        deconv_2 = self.stage_2_1(concat_2)
        
        decon_11 = F.interpolate(deconv_2, size=out_st1.size()[2:], mode='bilinear', align_corners=True)
        concat_1 = torch.cat([decon_11, out_st1], 1)
        
        out = self.stage_mid(concat_1)
        return out
        
        
class _enc_block(nn.Module):
    def __init__(self, enc_dep_set):
        super(_enc_block, self).__init__()
        
        enc1, enc2, enc3, enc4 = enc_dep_set[0], enc_dep_set[1], enc_dep_set[2], enc_dep_set[3]
        
        self.stage_1_1 = self.make_layer(_conv_block, [3,    enc1])
        self.stage_2_1 = self.make_layer(_conv_block, [enc1, enc2])
        self.stage_3_1 = self.make_layer(_conv_block, [enc2, enc3])
        self.stage_4_1 = self.make_layer(_conv_block, [enc3, enc4])
        
        self.maxpool = nn.MaxPool2d(2)
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        out1 = self.stage_1_1(x)
        out = self.maxpool(out1)
        
        out2 = self.stage_2_1(out)
        out = self.maxpool(out2)
        
        out3 = self.stage_3_1(out)
        out = self.maxpool(out3)
        
        out4 = self.stage_4_1(out)
        output = self.maxpool(out4)
        
        return output, [out1, out2, out3, out4]
    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
        self.vgg19 = self.make_layer(_vgg19, 0)
        
        self.aspp1 = ASPP(256,256)
        self.dec_1 = self.make_layer(_dec_block, [96, 192, 384, 512])

        self.enc_1 = self.make_layer(_enc_block, [32, 64, 128, 256])
        self.aspp2 = ASPP(256,256)
        self.dec_2 = self.make_layer(_dec_block, [128, 256, 512, 768])
        
        self.out_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.ReLU6()
        #self.output = nn.Sigmoid()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        residual = x
        
        # network part - 1
        out, skip1 = self.vgg19(x)        
        out_aspp = self.aspp1(out)
        out = self.dec_1([out_aspp, skip1])
        
        # bridge part
        output_1 = residual * out
        
        out, skip2 = self.enc_1(output_1)
        out_aspp= self.aspp2(out)
        
        # modification part
        stage_1_feat = torch.cat([skip1[0], skip2[0]], 1)
        stage_2_feat = torch.cat([skip1[1], skip2[1]], 1)
        stage_3_feat = torch.cat([skip1[2], skip2[2]], 1)
        stage_4_feat = torch.cat([skip1[3], skip2[3]], 1)
        
        skip_modif = [stage_1_feat, stage_2_feat, stage_3_feat, stage_4_feat]
        out = self.dec_2([out_aspp, skip_modif])
        
        output_2 = torch.cat([output_1, out], 1)
        out = self.out_conv(output_2)
        out = ( self.output(out) * (5/3) ) - ( 5 * torch.ones_like(out) )
        
        return out
