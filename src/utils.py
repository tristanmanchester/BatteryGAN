import torch
import numpy as np
from PIL import Image
import cv2

def load_image(path, grayscale=True):
    """
    Load an image from disk using OpenCV.
    Preserves original bit depth with IMREAD_UNCHANGED.
    Properly scales to 8-bit without clipping.
    If grayscale=True, returns a single-channel image.
    """
    # Load image with original bit depth preserved
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert BGR to RGB if color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Scale to 8-bit if higher bit depth
    if img.dtype == np.uint16:
        # Find actual min and max for proper scaling
        img_min, img_max = img.min(), img.max()
        # Scale to full 8-bit range without clipping
        if img_max > img_min:  # Avoid division by zero
            img = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    
    # Handle grayscale conversion if needed
    if grayscale:
        if len(img.shape) == 3:  # Color image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(img)
    else:
        # If grayscale image but RGB requested
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        return Image.fromarray(img)

def save_image(tensor, path, grayscale=True):
    """
    Clamp tensor to [0,1], convert to uint8 and save as image.
    If grayscale=True, assumes the tensor is a single-channel image [1,H,W].
    """
    tensor = tensor.clamp(0, 1)
    
    if grayscale:
        # For grayscale, tensor should be [1,H,W] or [H,W]
        if tensor.dim() == 3 and tensor.size(0) == 1:
            # Convert [1,H,W] to [H,W]
            ndarr = (tensor.mul(255)
                        .squeeze(0)
                        .byte()
                        .cpu()
                        .numpy())
        elif tensor.dim() == 3 and tensor.size(0) == 3:
            # Convert RGB to grayscale using standard coefficients
            tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            ndarr = (tensor.mul(255)
                        .byte()
                        .cpu()
                        .numpy())
        else:
            # Already [H,W]
            ndarr = (tensor.mul(255)
                        .byte()
                        .cpu()
                        .numpy())
    else:
        # RGB image [3,H,W]
        ndarr = (tensor.mul(255)
                    .permute(1, 2, 0)
                    .byte()
                    .cpu()
                    .numpy())
    
    if str(path).lower().endswith(('.tif', '.tiff')):
        # Save as TIFF using OpenCV
        cv2.imwrite(str(path), ndarr)
    else:
        # For other formats, use PIL
        img = Image.fromarray(ndarr)
        img.save(path)
