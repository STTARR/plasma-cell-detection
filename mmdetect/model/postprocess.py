import torch
import numpy as np
import scipy.ndimage
from skimage.feature import peak_local_max


def postprocess(out, median_sz, blur_sigma):
    """Apply postprocessing on multidimensional tensor
    in 2D (ignoring the batch and channel dimensions, i.e. postprocess each channel individually).
    Last two channels are assumed to be (h,w)"""
    extradims = out.ndim - 2
    h, w = out.shape[-2:]
    median = scipy.ndimage.median_filter(out, footprint=np.ones([1]*extradims + [median_sz, median_sz]))
    blurred = scipy.ndimage.gaussian_filter(median, [0]*extradims + [blur_sigma, blur_sigma])
    return blurred


def model_output_to_points(out, median=3, blur=3, min_distance=10, flood_fill_brown_into_blue=False):
    """Note returned values are in YX order (numpy order)."""
    brown_cells = out.argmax(dim=0) == 2
    if flood_fill_brown_into_blue:
        # Flood fill brown cells into blue cells if they're bordering (since it's a membrane stain this can be helpful)
        any_cells = out.argmax(dim=0) > 0
        brown_cells = torch.from_numpy(scipy.ndimage.binary_dilation(brown_cells, iterations=-1, mask=any_cells))
        blue_cells = any_cells ^ brown_cells
    else:
        blue_cells = out.argmax(dim=0) == 1
    brown_cells_yx = peak_local_max(postprocess(brown_cells.float() * out[2], median, blur), min_distance=min_distance, indices=True)
    blue_cells_yx = peak_local_max(postprocess(blue_cells.float() * out[2], median, blur), min_distance=min_distance, indices=True)
    return blue_cells_yx, brown_cells_yx
