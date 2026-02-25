
import torch

from copy import deepcopy

from sr.modeling.archs.swinir import SwinIR
from sr.modeling.archs.interp import Interpolate


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            dataset (dict): Dataset type and image shape configuration.
            network (dict): Model type and configuration.
    """

    # Retrieve input image size.
    opt['network']['img_size'] = opt['dataset']['input_shape']
    
    if opt['network']["name"] == "SwinIR":
        # Set super-resolution SwinIR network.
        model = SwinIR(**opt['network'])
        # Print model information (parameters, etc.).
        print_model_info(model)
        # Load pre-trained weights.
        print("Loading weights from:", opt['network']['weights_path'])
        model.load_from_state_dict(opt['network']['weights_path'])
    else:
        if opt['network']["name"] != "Interpolation":
            print("Model config not found. Using bicubic interpolation.")
        # Apply vanilla bicubic interpolation for super-resolution.
        
        model = Interpolate(scale_factor=opt['dataset']['scale_factor'])
        # Print model information (parameters, etc.).
        print_model_info(model)

    return model


def print_model_info(model):
    def compute_num_params(module):
        num_params = sum(p.numel() for p in module.parameters())
        return num_params
    
    print('Model Information:')
    print(f'Network parameters (M): {compute_num_params(model) / 1e6:.2f}')