import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ResUnetPlusPlus_BE(nn.Module):
    def __init__(self,  filters=[32, 64, 128, 256, 512], pretrained=False, mode = 'Train'):
        super(ResUnetPlusPlus_BE, self).__init__()
        self.mode = mode
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv2d(filters[0], 1, 1))

                 # HED Block
        self.dsn1 = nn.Conv2d(filters[0], 1, 1)
        self.dsn2 = nn.Conv2d(filters[1], 1, 1)
        self.dsn3 = nn.Conv2d(filters[2], 1, 1)
        self.dsn4 = nn.Conv2d(filters[3], 1, 1)
        self.dsn5 = nn.Conv2d(filters[4], 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(5, 32, 1),nn.ReLU(inplace=True))
#         self.fuse = nn.Conv2d(5, 64, 1)
        
        self.SE_mimic = nn.Sequential(        
            nn.Linear(32, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5, bias=False),
            nn.Sigmoid()
        )
        self.final_boundary = nn.Conv2d(5,2,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True)            
        )
        self.final_mask = nn.Conv2d(64,2,1)
        
                

        self.relu = nn.ReLU()        
        self.out = nn.Conv2d(64,1,1)



    def forward(self, x):
        h = x.size(2)
        w = x.size(3)
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        xx = self.aspp_out(x8)
#         out = self.output_layer(x9)
#         out = F.sigmoid(out)
                
         ## side output
        d1 = self.dsn1(x1)
        d2 = F.upsample_bilinear(self.dsn2(x2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(x3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(x4), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(x5), size=(h,w))
#
        ###########sigmoid ver
        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        d5_out = F.sigmoid(d5)

        concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out), 1)

        fuse_box = self.fuse(concat)
        GAP = F.adaptive_avg_pool2d(fuse_box,(1,1))
        GAP = GAP.view(-1, 32)        
        se_like = self.SE_mimic(GAP)        
        se_like = torch.unsqueeze(se_like, 2)
        se_like = torch.unsqueeze(se_like, 3)

        feat_se = concat * se_like.expand_as(concat)
        boundary = self.final_boundary(feat_se)
        boundary_out = torch.unsqueeze(boundary[:,1,:,:],1)        
        bd_sftmax = F.softmax(boundary, dim=1)
        boundary_scale = torch.unsqueeze(bd_sftmax[:,1,:,:],1)        
        
        feat_concat = torch.cat( [xx, fuse_box], 1)
        feat_concat_conv = self.final_conv(feat_concat)
        mask = self.final_mask(feat_concat_conv)
        mask_sftmax = F.softmax(mask,dim=1)        
        mask_scale = torch.unsqueeze(mask_sftmax[:,1,:,:],1)

        if self.mode == 'Train':
            scalefactor = torch.clamp(mask_scale+boundary_scale,0,1)            
        elif self.mode == 'Infer':
            scalefactor = torch.clamp(mask_scale+5*boundary_scale,0,1)
        
        
        mask_out = torch.unsqueeze(mask[:,1,:,:],1)
        relu = self.relu(mask_out)        
        scalar = relu.cpu().detach().numpy()
        if np.sum(scalar) == 0:
            average = 0
        else : 
            average = scalar[np.nonzero(scalar)].mean()
        mask_out = mask_out-relu + (average*scalefactor)
        
        if self.mode == 'Train':
            mask_out = F.sigmoid(mask_out)
            boundary_out = F.sigmoid(boundary_out)
            
            return d1_out, d2_out, d3_out, d4_out, d5_out, boundary_out, mask_out
        elif self.mode =='Infer':
            return mask_out
        
  
    
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2    