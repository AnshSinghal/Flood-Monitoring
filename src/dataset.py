import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from data_utils import (load_and_stack_sar, load_and_stack_optical, load_cloud_mask,
    apply_speckle_filter, normalize_sar, normalize_optical, get_cloud_coverage 
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class FloodDataset(Dataset):
    def __init__(self, manifest_path, split='train', cloud_threshold=10.0, use_speckle_filter=True):
        """
        Custom dataset for flood mapping using Sentinel-2 optical and Sentinel-1 SAR images.
        Args: 
            manifest_path (str): Path to the dataset manifest CSV file.
            split (str): Dataset split - 'train', 'val', or 'test'.
            cloud_threshold (float): Maximum cloud coverage percentage to include an image.
            use_speckle_filter (bool): Whether to apply speckle filtering to SAR images.
        """
        logger.info("Initialising FloodDataset with manifest=%s split=%s", manifest_path, split)
        self.manifest = pd.read_csv(manifest_path)
        self.split = split
        self.cloud_threshold = cloud_threshold
        self.use_speckle_filter = use_speckle_filter

        #Filter by split
        self.df = self.manifest[self.manifest['split'] == self.split].reset_index(drop=True)

        logger.debug("Loaded %s entries for split '%s' before cloud filtering", len(self.df), self.split)

        #For training, filter out images with high cloud coverage
        if self.split == 'train':
            is_clear = []
            for idx in range(len(self.df)):
                try:
                    mask = load_cloud_mask(self.df.iloc[idx]['s2_cloudmask'])
                    coverage = get_cloud_coverage(mask)
                    is_clear.append(coverage < self.cloud_threshold)
                except Exception:
                    logger.exception("Failed to load cloud mask for index %s", idx)
                    is_clear.append(False)

            self.df = self.df[is_clear].reset_index(drop=True)
            logger.info("[%s] Filtered to %s samples with < %.2f%% cloud cover.", split, len(self.df), self.cloud_threshold)

        if self.split == 'train':
            # For training, apply random flips for data augmentation
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ], additional_targets=['image0', 'image'])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ], additional_targets=['image0', 'image'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        logger.debug("Fetching item at index %s", idx)

        try:
            sar_img = load_and_stack_sar(row['sar_vv'], row['sar_vh'])
            opt_img = load_and_stack_optical(row['s2_b4_red'], row['s2_b3_green'], row['s2_b2_blue'])

            if self.use_speckle_filter:
                logger.debug("Applying speckle filter to SAR image at idx=%s", idx)
                sar_img = apply_speckle_filter(sar_img)

            sar_img = normalize_sar(sar_img)
            opt_img = normalize_optical(opt_img)

            # Albumentations expects channel-last format (H, W, C)
            sar_img = np.moveaxis(sar_img, 0, -1)  # Move channels to last dimension
            opt_img = np.moveaxis(opt_img, 0, -1)  # Move channels to last dimension

            # Apply augmentations (which also converts to tensor)
            augmented = self.transform(image=sar_img, image0=opt_img)
            sar_tensor = augmented['image']
            opt_tensor = augmented['image0']

            logger.debug("Successfully loaded item %s", idx)
            return sar_tensor, opt_tensor
        
        except Exception as e:
            logger.exception("Error loading data at index %s", idx)
            # Return None to allow DataLoader collate_fn to handle it
            return None