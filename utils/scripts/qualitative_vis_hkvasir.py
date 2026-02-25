import torch
import cv2
import numpy as np
import os


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

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img

def overlay_mask_rgb(image, mask, alpha=0.35, color=(0, 255, 255)):
    """
    Overlay a binary mask on an RGB image.

    Parameters
    ----------
    image : np.ndarray (uint8, RGB)
    mask  : np.ndarray (0/1 or bool)
    alpha : float (0–1)
    color : tuple in RGB (default: cyan)

    Returns
    -------
    overlay : np.ndarray
    """
    overlay = image.copy()
    mask = mask.astype(bool)

    # Create solid color image (RGB)
    color_img = np.zeros_like(image, dtype=np.uint8)
    color_img[:] = color

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, color_img, alpha, 0)

    # Replace only masked pixels
    overlay[mask] = blended[mask]

    return overlay

def add_contour_rgb(image, mask, color=(0,255,255), thickness=2):
    contour_img = image.copy()
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(contour_img, contours, -1, color, thickness)
    return contour_img

def remove_border(mask, border=6):
    mask = mask.copy()
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask

def crop_square(image, center, size=64):
    """
    Crop a square patch around a central coordinate.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image
    center : tuple
        (row, col) coordinate of the center
    size : int
        Side length of the square

    Returns
    -------
    patch : np.ndarray
        Cropped square image
    top_left : tuple
        Coordinates of top-left corner in the original image
    """
    h, w, _ = image.shape
    r, c = center
    half = size // 2

    # Compute bounds, clamp to image
    r1 = max(r - half, 0)
    r2 = min(r + half, h)
    c1 = max(c - half, 0)
    c2 = min(c + half, w)

    patch = image[r1:r2, c1:c2].copy()
    return patch, (r1, c1)

def draw_square(image, top_left, size=64, color=(255, 255, 255), thickness=2):
    """
    Draw a square rectangle on an image.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 RGB image (will be modified)
    top_left : tuple
        (row, col) coordinates of top-left corner
    size : int
        Side length of the square
    color : tuple
        RGB color of the rectangle
    thickness : int
        Line thickness

    Returns
    -------
    image_with_box : np.ndarray
        Image with the rectangle drawn
    """
    r1, c1 = top_left
    r2 = r1 + size
    c2 = c1 + size

    # Draw rectangle (cv2.rectangle expects (x, y) = (col, row))
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (c1, r1), (c2, r2), color, thickness)
    return image_with_box

def label_patch_multiline(patch, texts, text_color=(0,0,0),
                          font_scale=0.5, thickness=1,
                          top_margin=2, line_spacing=2,
                          bg_color=(255,255,255), alpha=0.6):
    """
    Add one or more lines of text at the top of a patch with a semi-transparent background.

    Parameters
    ----------
    patch : np.ndarray
        HxWx3 RGB patch
    texts : list of str
        Text lines to display
    text_color : tuple
        RGB color of text
    font_scale : float
        Font size scaling
    thickness : int
        Line thickness
    top_margin : int
        Pixels from top edge
    line_spacing : int
        Pixels between lines
    bg_color : tuple
        RGB background rectangle color
    alpha : float
        Transparency of the background (0=transparent, 1=solid)

    Returns
    -------
    labeled_patch : np.ndarray
        Patch with text and background rectangle
    """
    labeled_patch = patch.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compute total height of all text lines
    heights = []
    widths = []
    for text in texts:
        size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        widths.append(size[0])
        heights.append(size[1])

    total_height = sum(heights) + line_spacing * (len(texts) - 1)
    rect_height = top_margin + total_height + 2  # extra 2 px padding

    # Draw semi-transparent rectangle across full width
    overlay = labeled_patch.copy()
    cv2.rectangle(overlay, (0, 0), (patch.shape[1], rect_height), bg_color, -1)
    labeled_patch = cv2.addWeighted(overlay, alpha, labeled_patch, 1-alpha, 0)

    # Draw each text line
    y_cursor = top_margin
    for i, text in enumerate(texts):
        text_width = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        x_text = max((patch.shape[1] - text_width) // 2, 0)  # centered
        y_text = y_cursor + heights[i]
        cv2.putText(labeled_patch, text, (x_text, y_text), font,
                    font_scale, text_color, thickness, lineType=cv2.LINE_AA)
        y_cursor += heights[i] + line_spacing

    return labeled_patch

def add_panel_letter(patch, letter="A",
                     font_scale=0.6,
                     thickness=1,
                     margin=4):
    """
    Add a white panel letter with thin black outline
    in the lower-left corner.
    """
    labeled_patch = patch.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    h, w, _ = patch.shape
    x = margin
    y = h - margin

    # Black outline (draw thicker)
    cv2.putText(labeled_patch, letter, (x, y),
                font, font_scale, (0,0,0),
                thickness+2, lineType=cv2.LINE_AA)

    # White letter on top
    cv2.putText(labeled_patch, letter, (x, y),
                font, font_scale, (255,255,255),
                thickness, lineType=cv2.LINE_AA)

    return labeled_patch

def draw_box_with_letter(image, top_left, size=64,
                         letter="A",
                         box_color=(0,220,220),
                         box_thickness=2,
                         font_scale=1.0,
                         text_thickness=2,
                         offset=5):
    """
    Draw a square bounding box and place a letter near it.

    Parameters
    ----------
    image : np.ndarray (RGB)
    top_left : tuple (row, col)
    size : int
    letter : str
    box_color : tuple (RGB)
    box_thickness : int
    font_scale : float
    text_thickness : int
    offset : int
        Distance of letter from the box

    Returns
    -------
    annotated_image : np.ndarray
    """
    annotated = image.copy()

    r1, c1 = top_left
    r2 = r1 + size
    c2 = c1 + size

    # Draw bounding box
    cv2.rectangle(annotated, (c1, r1), (c2, r2),
                  box_color, box_thickness)

    # Position letter slightly above top-left corner
    x_text = c1
    y_text = max(r1 - offset, 10)

    # Black outline
    cv2.putText(annotated, letter, (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0,0,0),
                text_thickness+3,
                lineType=cv2.LINE_AA)

    # White letter
    cv2.putText(annotated, letter, (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255,255,255),
                text_thickness,
                lineType=cv2.LINE_AA)

    return annotated

# Path, folders, images ID
PATH_DATASETS = "/usr/bmicnas03/data-biwi-01/jusilva_data/data/datasets/Endoscopy/"
PATH_EXPERIMENTS = "./docs/local_data/experiments/"
PATH_PLOTS = "./docs/local_data/visualizations/hkvasir/"

# Create bounding boxes
box_size = 150

# -------------------------------------------------------------------
# Hyperkvasir dataset - Case 1
dataset = 'Hyperkvasir/'
sample = '0277/0049.png'

# Read actual image.
img_ref = cv2.imread(os.path.join(PATH_DATASETS, "Hyperkvasir/hyperkvasir_test/GT/", sample))
img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
# Read predicted image by model.
img_prd_model = cv2.imread(os.path.join(PATH_EXPERIMENTS, dataset, "BasicVSR/Preds", sample.replace('/','/0000')))
img_prd_model = cv2.cvtColor(img_prd_model, cv2.COLOR_BGR2RGB)
# Read predicted error map.
imf_cfm_model = cv2.imread(os.path.join(PATH_EXPERIMENTS, dataset, "BasicVSR/Errornet-2layer/CRCMask", sample.replace('/','/0000')))
imf_cfm_model = cv2.cvtColor(imf_cfm_model, cv2.COLOR_BGR2GRAY) == 255
imf_cfm_model = remove_border(imf_cfm_model, border=6)

# Post-process mask to remove landmarks non related to image
imf_cfm_model[0:300, 0:200] = 0
imf_cfm_model[0:50, 0:500] = 0

# Created overlaid image
overlay = overlay_mask_rgb(img_prd_model, imf_cfm_model, alpha=0.28, color=(0, 220, 220))
overlay = add_contour_rgb(overlay, imf_cfm_model, color=(0,220,220), thickness=1)

# Bounding box A.
center = (130, 350)  # y, x
patch_A_img, top_left = crop_square(img_ref, center, size=box_size)
patch_A_model, top_left = crop_square(img_prd_model, center, size=box_size)
psnr_A, _ = calculate_psnr(torch.tensor(patch_A_img).permute(2, 0, 1).unsqueeze(0) / 255,
                           torch.tensor(patch_A_model).permute(2, 0, 1).unsqueeze(0) / 255)

overlay = draw_box_with_letter(overlay, top_left, size=box_size, letter="E", box_color=(255,220,255), box_thickness=2, font_scale=1.2, text_thickness=2, offset=6)
patch_A_img = label_patch_multiline(patch_A_img, texts=["High Resolution"], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_A_img = add_panel_letter(patch_A_img, letter="E", font_scale=0.6, thickness=1, margin=4)
patch_A_model = label_patch_multiline(patch_A_model, texts=["BasicVSR", "PSNR:" + str(np.round(psnr_A, 1)) + 'dB'], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_A_model = add_panel_letter(patch_A_model, letter="E", font_scale=0.6, thickness=1, margin=4)

# Bounding box B.
center = (330, 200)  # y, x
patch_B_img, top_left = crop_square(img_ref, center, size=box_size)
patch_B_model, top_left = crop_square(img_prd_model, center, size=box_size)
psnr_B, _ = calculate_psnr(torch.tensor(patch_B_img).permute(2, 0, 1).unsqueeze(0) / 255,
                           torch.tensor(patch_B_model).permute(2, 0, 1).unsqueeze(0) / 255)
overlay = draw_box_with_letter(overlay, top_left, size=box_size, letter="F", box_color=(255,220,255), box_thickness=2, font_scale=1.2, text_thickness=2, offset=6)
patch_B_img = label_patch_multiline(patch_B_img, texts=["High Resolution"], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_B_img = add_panel_letter(patch_B_img, letter="F", font_scale=0.6, thickness=1, margin=4)
patch_B_model = label_patch_multiline(patch_B_model, texts=["BasicVSR", "PSNR:" + str(np.round(psnr_B, 1)) + 'dB'], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_B_model = add_panel_letter(patch_B_model, letter="F", font_scale=0.6, thickness=1, margin=4)

cv2.imwrite(PATH_PLOTS + "image_1_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_1_patch_A_img.png", cv2.cvtColor(patch_A_img, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_1_patch_A_basicvsr.png", cv2.cvtColor(patch_A_model, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_1_patch_B_img.png", cv2.cvtColor(patch_B_img, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_1_patch_B_basicvsr.png", cv2.cvtColor(patch_B_model, cv2.COLOR_RGB2BGR).astype(np.uint8))

# -------------------------------------------------------------------
# Hyperkvasir dataset - Case 2
dataset = 'Hyperkvasir/'
sample = '0258/0045.png'

# Read actual image.
img_ref = cv2.imread(os.path.join(PATH_DATASETS, "Hyperkvasir/hyperkvasir_test/GT/", sample))
img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
# Read predicted image by model.
img_prd_model = cv2.imread(os.path.join(PATH_EXPERIMENTS, dataset, "BasicVSR/Preds", sample.replace('/','/0000')))
img_prd_model = cv2.cvtColor(img_prd_model, cv2.COLOR_BGR2RGB)
# Read predicted error map.
imf_cfm_model = cv2.imread(os.path.join(PATH_EXPERIMENTS, dataset, "BasicVSR/Errornet-2layer/CRCMask", sample.replace('/','/0000')))
imf_cfm_model = cv2.cvtColor(imf_cfm_model, cv2.COLOR_BGR2GRAY) == 255
imf_cfm_model = remove_border(imf_cfm_model, border=6)

# Post-process mask to remove landmarks non related to image
imf_cfm_model[0:150, 0:120] = 0
imf_cfm_model[0:70, 0:300] = 0
imf_cfm_model[-200:, 0:245] = 0
imf_cfm_model[-200:, -100:] = 0

# Created overlaid image
overlay = overlay_mask_rgb(img_prd_model, imf_cfm_model, alpha=0.28, color=(0, 220, 220))
overlay = add_contour_rgb(overlay, imf_cfm_model, color=(0,220,220), thickness=1)

# Bounding box A.
center = (150, 420)  # y, x
patch_A_img, top_left = crop_square(img_ref, center, size=box_size)
patch_A_model, top_left = crop_square(img_prd_model, center, size=box_size)
psnr_A, _ = calculate_psnr(torch.tensor(patch_A_img).permute(2, 0, 1).unsqueeze(0) / 255,
                           torch.tensor(patch_A_model).permute(2, 0, 1).unsqueeze(0) / 255)
overlay = draw_box_with_letter(overlay, top_left, size=box_size, letter="G", box_color=(255,220,255), box_thickness=2, font_scale=1.2, text_thickness=2, offset=6)
patch_A_img = label_patch_multiline(patch_A_img, texts=["High Resolution"], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_A_img = add_panel_letter(patch_A_img, letter="G", font_scale=0.6, thickness=1, margin=4)
patch_A_model = label_patch_multiline(patch_A_model, texts=["BasicVSR", "PSNR:" + str(np.round(psnr_A, 1)) + 'dB'], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_A_model = add_panel_letter(patch_A_model, letter="G", font_scale=0.6, thickness=1, margin=4)

# Bounding box B.
center = (430, 340)  # y, x
patch_B_img, top_left = crop_square(img_ref, center, size=box_size)
patch_B_model, top_left = crop_square(img_prd_model, center, size=box_size)
psnr_B, _ = calculate_psnr(torch.tensor(patch_B_img).permute(2, 0, 1).unsqueeze(0) / 255,
                           torch.tensor(patch_B_model).permute(2, 0, 1).unsqueeze(0) / 255)
overlay = draw_box_with_letter(overlay, top_left, size=box_size, letter="H", box_color=(255,220,255), box_thickness=2, font_scale=1.2, text_thickness=2, offset=6)
patch_B_img = label_patch_multiline(patch_B_img, texts=["High Resolution"], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_B_img = add_panel_letter(patch_B_img, letter="H", font_scale=0.6, thickness=1, margin=4)
patch_B_model = label_patch_multiline(patch_B_model, texts=["BasicVSR", "PSNR:" + str(np.round(psnr_B, 1)) + 'dB'], text_color=(0,0,0), font_scale=0.6, thickness=1, top_margin=5, alpha=0.2)
patch_B_model = add_panel_letter(patch_B_model, letter="H", font_scale=0.6, thickness=1, margin=4)

cv2.imwrite(PATH_PLOTS + "image_2_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_2_patch_A_img.png", cv2.cvtColor(patch_A_img, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_2_patch_A_basicvsr.png", cv2.cvtColor(patch_A_model, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_2_patch_B_img.png", cv2.cvtColor(patch_B_img, cv2.COLOR_RGB2BGR).astype(np.uint8))
cv2.imwrite(PATH_PLOTS + "image_2_patch_B_basicvsr.png", cv2.cvtColor(patch_B_model, cv2.COLOR_RGB2BGR).astype(np.uint8))
