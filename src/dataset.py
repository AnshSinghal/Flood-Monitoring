import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from data_utils import (load_and_stack_sar, load_and_stack_optical, load_cloud_mask,
    apply_speckle_filter, normalize_sar, normalize_optical, get_cloud_coverage 
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        self.manifest = pd.read_csv(manifest_path)
        self.split = split
        self.cloud_threshold = cloud_threshold
        self.use_speckle_filter = use_speckle_filter

        #Filter by split
        self.df = self.manifest[self.manifest['split'] == self.split].reset_index(drop=True)

        #For training, filter out images with high cloud coverage
        if self.split == 'train':
            is_clear = []
            for idx in range(len(self.df)):
                mask = load_cloud_mask(self.df.iloc[idx]['s2_cloudmask'])
                coverage = get_cloud_coverage(mask)
                is_clear.append(coverage < self.cloud_threshold)

            self.df = self.df[is_clear].reset_index(drop=True)
            print(f"[{split}] Filtered to {len(self.df)} samples with < {self.cloud_threshold}% cloud cover.")

        if self.split == 'train':
            # For training, apply random flips for data augmentation
            self.trabsform = A.Compose([
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

        try:
            sar_img = load_and_stack_sar(row['sar_vv'], row['sar_vh'])
            opt_img = load_and_stack_optical(row['s2_b4_red'], row['s2_b3_green'], row['s2_b2_blue'])

            if self.use_speckle_filter:
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

            return sar_tensor, opt_tensor
        
        except Exception as e:
            print(f"Error loading data at index {idx}, path: {row['s1_vv']}")
            print(f"Error: {e}")
            # Return None or a placeholder if you want to handle errors gracefully in the DataLoader
            return None