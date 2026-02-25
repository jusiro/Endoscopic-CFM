import torch

class Interpolate(torch.nn.Module):
    def __init__(self, scale_factor, mode='bicubic', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)