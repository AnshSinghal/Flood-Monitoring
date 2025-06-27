import numpy as np
import rasterio
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def lee_filter(img, size):
    logger.debug("Applying Lee filter with window size=%s", size)
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def load_and_stack_sar(vv_path, vh_path):
    logger.debug("Loading SAR VV from %s and VH from %s", vv_path, vh_path)
    try:
        with rasterio.open(vv_path) as src:
            s1_vv = src.read(1).astype(np.float32)
        with rasterio.open(vh_path) as src:
            s1_vh = src.read(1).astype(np.float32)
    except Exception:
        logger.exception("Failed to load SAR images: VV=%s VH=%s", vv_path, vh_path)
        raise

    # Replace nodata values (often 0 in this dataset) with a very small number for log conversion
    s1_vv[s1_vv == 0] = 1e-6
    s1_vh[s1_vh == 0] = 1e-6

    # Stack the VV and VH bands
    return np.stack([s1_vv, s1_vh], axis=0)

def load_and_stack_optical(r_path, g_path, b_path):
    logger.debug("Loading optical R=%s G=%s B=%s", r_path, g_path, b_path)
    try:
        with rasterio.open(r_path) as src:
            s2_r = src.read(1).astype(np.float32)
        with rasterio.open(g_path) as src:
            s2_g = src.read(1).astype(np.float32)
        with rasterio.open(b_path) as src:
            s2_b = src.read(1).astype(np.float32)
    except Exception:
        logger.exception("Failed to load optical images")
        raise
    return np.stack([s2_r, s2_g, s2_b], axis=0)

def load_cloud_mask(mask_path):
    logger.debug("Loading cloud mask from %s", mask_path)
    try:
        with rasterio.open(mask_path) as src:
            cloud_mask = src.read(1).astype(np.uint8)
    except Exception:
        logger.exception("Failed to load cloud mask %s", mask_path)
        raise
    return cloud_mask

def apply_speckle_filter(sar_image, window_size=7):
    logger.debug("Applying speckle filter with window_size=%s on SAR image", window_size)
    try:
        filtered_sar = []
        for i in range(sar_image.shape[0]):
            filtered_sar.append(lee_filter(sar_image[i], size=window_size))
        return np.stack(filtered_sar, axis=0)
    except Exception:
        logger.exception("Speckle filtering failed")
        raise

def normalize_sar(sar_img):
    logger.debug("Normalising SAR image to [-1,1] range")
    db_image = 10 * np.log10(sar_img) # Convert the linear SAR image to dB

    min_db, max_db = -25.0, 0.0
    db_img = np.clip(db_image, min_db, max_db)

    normalized_image = 2 * ((db_img - min_db) / (max_db - min_db)) - 1 # Normalize to [-1, 1]
    return normalized_image.astype(np.float32)

def normalize_optical(optical_img):
    logger.debug("Normalising optical image to [-1,1] range")
    clip_max = 4000.0
    opt_img = np.clip(optical_img, 0, clip_max) # Clip to avoid extreme values
    normalized_image = 2 * (opt_img / clip_max) - 1 # Normalize to [-1, 1]
    return normalized_image.astype(np.float32)

def get_cloud_coverage(cloud_mask):
    coverage = np.sum(cloud_mask == 1) / cloud_mask.size * 100 if cloud_mask.size > 0 else 0
    logger.debug("Computed cloud coverage: %.2f%%", coverage)
    return coverage
