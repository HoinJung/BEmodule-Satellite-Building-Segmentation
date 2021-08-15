import numpy as np
from .torch_callbacks import torch_callback_dict
import torch


def get_callbacks(framework, config):
    callbacks = []
    if framework == 'torch':
        for callback, params in config['training']['callbacks'].items():
            if callback == 'lr_schedule':
                callbacks.append(get_lr_schedule(framework, config))
            else:
                callbacks.append(torch_callback_dict[callback](**params))

    return callbacks


def get_lr_schedule(framework, config):
    

    schedule_type = config['training'][
        'callbacks']['lr_schedule']['schedule_type']
    initial_lr = config['training']['lr']
    update_frequency = config['training']['callbacks']['lr_schedule'].get(
        'update_frequency', 1)
    factor = config['training']['callbacks']['lr_schedule'].get(
        'factor', 0)
    schedule_dict = config['training']['callbacks']['lr_schedule'].get(
        'schedule_dict', None)
    if framework == 'torch':
        # just get the class itself to use; don't instantiate until the
        # optimizer has been created.
        if config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'linear':
            lr_scheduler = torch.optim.lr_scheduler.StepLR
        elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
#        elif config['training'][
#                'callbacks']['lr_schedule']['schedule_type'] == 'arbitrary':
#            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
        elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'arbitrary':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
            
        elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'cycle':
            print("check callback")
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR

            
    return lr_scheduler

