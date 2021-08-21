import os
from .. import weights_dir

from .unet import UNet
from .unet_BE import UNet_BE
from .resunet import ResUnetPlusPlus
from .resunet_BE import ResUnetPlusPlus_BE
from .ternaus import ternaus11
from .ternaus_BE import ternaus_BE
from .uspp import Uspp
from .uspp_BE import Uspp_BE
from .denet import DeNet
from .brrnet import BRRNet
from .brrnet_BE import BRRNet_BE
from .enru import ENRUNet
from .enru_BE import ENRUNet_BE

model_dict = {
    'unet' : {
        'weight_path': None,
        'weight_url': None,
        'arch': UNet},
    'enru' : {
        'weight_path': None,
        'weight_url': None,
        'arch': ENRUNet},
    'enru_BE' : {
        'weight_path': None,
        'weight_url': None,
        'arch': ENRUNet_BE},
    'brrnet' : {
        'weight_path': None,
        'weight_url': None,
        'arch': BRRNet},
    'brrnet_BE' : {
        'weight_path': None,
        'weight_url': None,
        'arch': BRRNet_BE},
    'denet' : {
        'weight_path': None,
        'weight_url': None,
        'arch': DeNet},
    'uspp' : {
        'weight_path': None,
        'weight_url': None,
        'arch': Uspp},
    'uspp_BE' : {
        'weight_path': None,
        'weight_url': None,
        'arch': Uspp_BE},    
    'resunet_BE' : {
        'weight_path':None,
        'weight_url': None,
        'arch': ResUnetPlusPlus_BE},
    'resunet' : {
        'weight_path': None,
        'weight_url': None,
        'arch': ResUnetPlusPlus},
    'unet_BE' : {
        'weight_path':None,
        'weight_url': None,
        'arch': UNet_BE},
    'ternaus' : {
        'weight_path':None,
        'weight_url': None,
        'arch': ternaus11},
    'ternaus_BE' : {
        'weight_path':None,
        'weight_url': None,
        'arch': ternaus_BE},
    'hrnetv2' : {
        'weight_path':None,
        'weight_url': None,
        'arch': hrnetv2},
    'hrnetv2_BE' : {
        'weight_path':None,
        'weight_url': None,
        'arch': hrnetv2_BE},
}
