import numpy as np

from ._torch_losses import torch_losses
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch
import skimage


def assembly_block(mask_64, mask):
    
    # obtain boundray from 1-channel mask
    arr_mask = mask.cpu().detach().numpy()
    mask_boundary_arr =  skimage.segmentation.find_boundaries(arr_mask, mode='inner', background=0).astype(np.float32)
    mask_boundary = torch.from_numpy(mask_boundary_arr).cuda().float()
    
    
    # recall 64-chanel mask before final conv
#     mask_boundary_arr =  skimage.segmentation.find_boundaries(mask_64, mode='inner', background=0).astype(np.float32)
    conv1 = nn.Conv2d(1,64,3,padding=1).cuda()
    conv2 = nn.Conv2d(128,64,3,padding=1).cuda()
    conv3 = nn.Conv2d(64,1,3,padding=1).cuda()
    
    x = Variable(mask_boundary, requires_grad=True)
    x = conv1(mask_boundary)
    x = torch.cat([mask_64, x], dim=1)
    x = conv2(x)
    x = conv3(x)

    return mask_boundary, x
            