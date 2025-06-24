import numpy as np
import rasterio
from skimage.filters import lee

def load_and_stack_sar(vv_path, vh_path):
    '''
    Load and stack SAR images (VV and VH bands).
    Args:
        vv_path (str): Path to the VV band image.
        vh_path (str): Path to the VH band image.

    Returns:
        np.ndarray: Stacked SAR image with shape (2, height, width).
    '''
    with rasterio.open(vv_path) as src:
        s1_vv = src.read(1).astype(np.float32)
    with rasterio.open(vh_path) as src:
        s1_vh = src.read(1).astype(np.float32)

    # Replace nodata values (often 0 in this dataset) with a very small number for log conversion
    s1_vv[s1_vv == 0] = 1e-6
    s1_vh[s1_vh == 0] = 1e-6

    # Stack the VV and VH bands
    return np.stack([s1_vv, s1_vh], axis=0)

def load_and_stack_optical(r_path, g_path, b_path):
    '''
    Load and stack optical images (Red, Green, Blue bands).
    Args:
        r_path (str): Path to the Red band image.
        g_path (str): Path to the Green band image.
        b_path (str): Path to the Blue band image.

    Returns:
        np.ndarray: Stacked optical image with shape (3, height, width).
    '''
    with rasterio.open(r_path) as src:
        s2_r = src.read(1).astype(np.float32)
    with rasterio.open(g_path) as src:
        s2_g = src.read(1).astype(np.float32)
    with rasterio.open(b_path) as src:
        s2_b = src.read(1).astype(np.float32)

    return np.stack([s2_r, s2_g, s2_b], axis=0)

def load_cloud_mask(mask_path):
    '''
    Load cloud mask image.
    Args:
        mask_path (str): Path to the cloud mask image.

    Returns:
        np.ndarray: Cloud mask with shape (height, width).
    '''
    with rasterio.open(mask_path) as src:
        cloud_mask = src.read(1).astype(np.uint8)

    return cloud_mask

def apply_speckle_filter(sar_image, window_size=7):
    '''
    Apply Lee speckle filter to the SAR image.
    Args:
        sar_image (np.ndarray): SAR image with shape (2, height, width).
        window_size (int): Size of the filtering window.
    Returns:
        np.ndarray: Filtered SAR image with shape (2, height, width).
    '''
    filtered_sar = []
    for i in range(sar_image.shape[0]):
        filtered_sar.append(lee(sar_image[i], M=window_size))
    return np.stack(filtered_sar, axis=0)

def normalize_sar(sar_img):
    """
    Converts linear SAR to dB, clips, and normalizes to [-1, 1].
    A common range for SAR backscatter is -25 to 0 dB.
    """
    db_image = 10 * np.log10(sar_img) # Convert the linear SAR image to dB

    min_db, max_db = -25.0, 0.0
    db_img = np.clip(db_image, min_db, max_db)

    normalized_image = 2 * ((db_img - min_db) / (max_db - min_db)) - 1 # Normalize to [-1, 1]
    return normalized_image.astype(np.float32)

def normalize_optical(optical_img):
    """
    Normalizes optical reflectance values (0-10000 range for S2 L1C) to [-1, 1].
    We clip at a lower value (e.g., 4000) to avoid extreme saturation from clouds/sun glint.
    """
    clip_max = 4000.0
    opt_img = np.clip(optical_img, 0, clip_max) # Clip to avoid extreme values
    normalized_image = 2 * (opt_img / clip_max) - 1 # Normalize to [-1, 1]
    return normalized_image.astype(np.float32)

def get_cloud_coverage(cloud_mask):
    """
    Calculates the percentage of cloudy pixels in a mask.
    In the C2S-MS dataset, 1 = Cloud. [9]
    """
    cloud_pixels = np.sum(cloud_mask == 1)
    total_pixels = cloud_mask.size
    return cloud_pixels / total_pixels * 100 if total_pixels > 0 else 0
