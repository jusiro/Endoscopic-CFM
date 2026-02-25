import cv2
import numpy as np
import torch

from sr.metrics.utils import rgb2ycbcr_pt

def calculate_psnr(img, img2, crop_border=0, test_y_channel=True, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)

    # Compute mse.
    mse_map = torch.mean((img - img2)**2, dim=1)
    mse = torch.mean(mse_map, dim=[1, 2])
    
    # Compute psnr.
    psnr = 10. * torch.log10(1. / (mse + 1e-8))
    return psnr.item(), mse_map.cpu().squeeze()