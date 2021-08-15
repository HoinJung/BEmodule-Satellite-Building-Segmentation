import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np

def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ternaus_BE(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False, mode= 'Train') -> None:
        """
        Args:
            num_filters:
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.conv6 = ConvRelu(num_filters * 8 * 2, num_filters * 8 * 2)
        self.decoder6 = nn.ConvTranspose2d(num_filters * 8 * 2,
                num_filters * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,)
        
        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

                 # HED Block
        self.dsn1 = nn.Conv2d(num_filters*2, 1, 1)
        self.dsn2 = nn.Conv2d(num_filters*4, 1, 1)
        self.dsn3 = nn.Conv2d(num_filters*8, 1, 1)
        self.dsn4 = nn.Conv2d(num_filters*16, 1, 1)
        self.dsn5 = nn.Conv2d(num_filters*16, 1, 1)
        self.dsn6 = nn.Conv2d(num_filters*8, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(6, 32, 1),nn.ReLU(inplace=True))
#         self.fuse = nn.Conv2d(5, 64, 1)
        
        self.SE_mimic = nn.Sequential(        
            nn.Linear(32, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6, bias=False),
            nn.Sigmoid()
        )
        self.final_boundary = nn.Conv2d(6,2,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64,64,3, padding=1),
            nn.ReLU(inplace=True)            
        )
        self.final_mask = nn.Conv2d(64,2,1)
        
                

        self.relu = nn.ReLU()        
        self.out = nn.Conv2d(64,1,1)
  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.size(2)
        w = x.size(3)
        conv1 = self.relu(self.conv1(x))
        conv1p = self.pool(conv1)
        conv2 = self.relu(self.conv2(conv1p))
        conv2p = self.pool(conv2)
        conv3s = self.relu(self.conv3s(conv2p))
        conv3 = self.relu(self.conv3(conv3s))
        conv3p = self.pool(conv3)
        conv4s = self.relu(self.conv4s(conv3p))
        conv4 = self.relu(self.conv4(conv4s))
        conv4p = self.pool(conv4)
        conv5s = self.relu(self.conv5s(conv4p))
        conv5 = self.relu(self.conv5(conv5s))
        conv5p = self.pool(conv5)
        
#         center = self.center(conv5p)
        conv6s = self.conv6(conv5p)
        conv6 = self.relu(self.decoder6(conv6s))
        center = conv6
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        xx = dec1
#         xx = self.final(dec1)
#         out = F.sigmoid(out)
                        
         ## side output
        d1 = self.dsn1(conv1)
        d2 = F.upsample_bilinear(self.dsn2(conv2), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(conv3), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(conv4), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(conv5), size=(h,w))
        
        d6 = F.upsample_bilinear(self.dsn6(conv6), size=(h,w))
#
        ###########sigmoid ver
        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        d5_out = F.sigmoid(d5)
        d6_out = F.sigmoid(d6)

#         concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out), 1)
        concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out,d6_out ), 1)

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
            
            return d1_out, d2_out, d3_out, d4_out, d5_out, d6_out,  boundary_out, mask_out
        elif self.mode =='Infer':
            return mask_out
        
        return out


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DecoderBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)