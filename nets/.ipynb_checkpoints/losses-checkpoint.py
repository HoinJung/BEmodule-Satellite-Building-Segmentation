import numpy as np
from ._torch_losses import torch_losses
from torch import nn


def get_loss(framework, loss, loss_weights=None, custom_losses=None):

    # lots of exception handling here. TODO: Refactor.
    
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:

        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        if framework in ['pytorch', 'torch']:
            return TorchCompositeLoss(loss, weights, custom_losses)

    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(framework, loss_name, loss_dict, custom_losses)


def get_single_loss(framework, loss_name, params_dict, custom_losses=None):

    if framework in ['torch', 'pytorch']:
        if params_dict is None:
            if custom_losses is not None and loss_name in custom_losses:
                return custom_losses.get(loss_name)()
            else:
                return torch_losses.get(loss_name.lower())()
        else:
            if custom_losses is not None and loss_name in custom_losses:
                return custom_losses.get(loss_name)(**params_dict)
            else:
                return torch_losses.get(loss_name.lower())(**params_dict)


class TorchCompositeLoss(nn.Module):
    """Composite loss function."""

    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss('pytorch',
                                                  loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]

        return loss
