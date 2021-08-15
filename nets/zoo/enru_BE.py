import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = ModuleHelper.BatchNorm2d(norm_type=norm_type)(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False, norm_type=None):
        super(ResNet, self).__init__()
        self.inplanes = 128 if deep_base else 16
        if deep_base:
            self.prefix = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('bn1', bn(64)),
                ('relu1', nn.ReLU(inplace=False)),
                ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn2', bn(64)),
                ('relu2', nn.ReLU(inplace=False)),
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
                ('bn3', bn(self.inplanes)),
                ('relu3', nn.ReLU(inplace=False))]
            ))
        else:
            self.prefix = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', bn(self.inplanes)),
                ('relu', nn.ReLU(inplace=False))]
            ))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change.

        self.layer1 = self._make_layer(block, 16, layers[0], norm_type=norm_type)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, norm_type=norm_type)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, ModuleHelper.BatchNorm2d(norm_type=norm_type, ret_cls=True)):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, norm_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_type=norm_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
class NormalResnetBackbone(nn.Module):
    def __init__(self, orig_resnet):
        super(NormalResnetBackbone, self).__init__()

        self.num_features = 512
        # take pretrained resnet, except AvgPool and FC
        self.prefix = orig_resnet.prefix
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def get_num_features(self):
        return self.num_features

    def forward(self, x):
        tuple_features = list()
        x = self.prefix(x)
        x = self.maxpool(x)
        x0 = x
        x1 = self.layer1(x)
        tuple_features.append(x1)
        
        x2 = self.layer2(x1)
        tuple_features.append(x2)
        x3 = self.layer3(x2)
        tuple_features.append(x3)
        x4 = self.layer4(x3)
        tuple_features.append(x4)

        return x0, x1, x2, x3, x4
    
def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_base=False, **kwargs)

    return model

def bn(num_features):
    return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU()
    )

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1,psp_size=(1,3,6,8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            bn(self.key_channels),
#             ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.psp(self.f_value(x))

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1,psp_size=(1,3,6,8)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                 
                                                   psp_size=psp_size)


class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.stages = []
        
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
#             ModuleHelper.BNReLU(out_channels, norm_type=norm_type),
            bn(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output
    

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
        
        
class ENRUNet_BE(nn.Sequential):
    def __init__(self,pretrained=False, mode='Train'):
        super(ENRUNet_BE, self).__init__()
        self.mode = mode
        self.backbone = NormalResnetBackbone(resnet50())
        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        self.dconv_up4 = double_conv(512+256, 256)
        self.dconv_up3 = double_conv(256+128, 128)
        self.dconv_up2 = double_conv(128+64, 64)
        self.dconv_up1 = double_conv(64 + 16, 64)
        self.APNB = nn.Sequential(
            APNB(in_channels=64, out_channels=64, key_channels=32, value_channels=32,
                         dropout=0.05, sizes=([1]))
        )
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        self.dsn1 = nn.Conv2d(16, 1, 1)
        self.dsn2 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(128, 1, 1)
        self.dsn4 = nn.Conv2d(256, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)

        #boundary enhancement part        
        self.fuse = nn.Sequential(nn.Conv2d(5, 64, 1),nn.ReLU(inplace=True))
#         self.fuse = nn.Conv2d(5, 64, 1)
        
        self.SE_mimic = nn.Sequential(        
            nn.Linear(64, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5, bias=False),
            nn.Sigmoid()
        )
        self.final_boundary = nn.Conv2d(5,2,1)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(128,64,3, padding=1),
            nn.ReLU(inplace=True)            
        )
        self.final_mask = nn.Conv2d(64,2,1)
        
                

        self.relu = nn.ReLU()        
        self.out = nn.Conv2d(64,1,1)

    def forward(self, x_):
        h = x_.size(2)
        w = x_.size(3)
        x0, x1, x2, x3, x4 = self.backbone(x_)
        up4 = F.interpolate(x4, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([up4, x3], dim=1)       
        x = self.dconv_up4(x)
        up3 = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([up3, x2], dim=1)       
        x = self.dconv_up3(x)
        up2 = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([up2, x1], dim=1)       
        x = self.dconv_up2(x)
        up1 = F.interpolate(x, size=(x0.size(2), x0.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([up1, x0], dim=1)       
        x = self.dconv_up1(x)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = self.APNB(x)
#         out = self.conv_last(x)
 
        ## side output
        d1 = F.upsample_bilinear(self.dsn1(x0), size=(h,w))
        d2 = F.upsample_bilinear(self.dsn2(x1), size=(h,w))
        d3 = F.upsample_bilinear(self.dsn3(x2), size=(h,w))
        d4 = F.upsample_bilinear(self.dsn4(x3), size=(h,w))
        d5 = F.upsample_bilinear(self.dsn5(x4), size=(h,w))

        ###########sigmoid ver
        d1_out = F.sigmoid(d1)
        d2_out = F.sigmoid(d2)
        d3_out = F.sigmoid(d3)
        d4_out = F.sigmoid(d4)
        d5_out = F.sigmoid(d5)

        concat = torch.cat((d1_out, d2_out, d3_out, d4_out, d5_out), 1)

              
        fuse_box = self.fuse(concat)
        GAP = F.adaptive_avg_pool2d(fuse_box,(1,1))
        GAP = GAP.view(-1, 64)        
        se_like = self.SE_mimic(GAP)        
        se_like = torch.unsqueeze(se_like, 2)
        se_like = torch.unsqueeze(se_like, 3)

        feat_se = concat * se_like.expand_as(concat)
        boundary = self.final_boundary(feat_se)
        boundary_out = torch.unsqueeze(boundary[:,1,:,:],1)        
        bd_sftmax = F.softmax(boundary, dim=1)
        boundary_scale = torch.unsqueeze(bd_sftmax[:,1,:,:],1)        
        
        feat_concat = torch.cat( [out, fuse_box], 1)
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