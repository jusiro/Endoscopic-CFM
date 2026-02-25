
import torch

from copy import deepcopy

from uq.errornet.modeling.archs.errornet import ErrorNet


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            network (dict): Model type and configuration.
    """

    # Set network.
    model = ErrorNet(**opt['errornet'])
    
    # Print model information (parameters, etc.).
    print_model_info(model)

    return model


def print_model_info(model):
    def compute_num_params(module):
        num_params = sum(p.numel() for p in module.parameters())
        return num_params
    
    print('Model Information:')
    print(f'Network parameters (M): {compute_num_params(model) / 1e6:.2f}')