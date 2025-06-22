import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_manifest_creation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def create_dataset_manifest(data_root, output_csv, test_size=0.15, random_state=42):
    """
    Parses the C2S-MS Floods dataset structure, correctly pairs S1 and S2 chips based on the
    actual file naming convention, and creates a manifest file with train/val splits.

    Args:
        data_root (str): Root directory where the event folders (UUIDs) are located.
        output_csv (str): Path to save the output CSV manifest file.
        test_size (float): The proportion of the dataset to allocate to the validation set.
        random_state (int): Seed for the random split for reproducibility.
    """
    try:
        event_dirs = [d for d in glob(os.path.join(data_root, '*')) if os.path.isdir(d)]
        if not event_dirs:
            logger.error(f"No event directories found in {data_root}. Please check the path.")
            return
        logger.info(f"Found {len(event_dirs)} event directories. Starting scan...")

        records = []
        for event_dir in tqdm(event_dirs, desc="Processing events"):
            s1_chip_dirs = sorted(glob(os.path.join(event_dir, 's1', '*')))
            s2_chip_dirs = sorted(glob(os.path.join(event_dir, 's2', '*')))

            if len(s1_chip_dirs) != len(s2_chip_dirs):
                logger.warning(f"Mismatch in number of chips for event {os.path.basename(event_dir)}: "
                               f"{len(s1_chip_dirs)} S1 chips vs {len(s2_chip_dirs)} S2 chips. Skipping event.")
                continue

            for s1_chip_dir, s2_chip_dir in zip(s1_chip_dirs, s2_chip_dirs):
                s1_vv_path = os.path.join(s1_chip_dir, 'VV.tif') 
                s1_vh_path = os.path.join(s1_chip_dir, 'VH.tif') 
                
                s2_b4_path = os.path.join(s2_chip_dir, 'B4.tif') 
                s2_b3_path = os.path.join(s2_chip_dir, 'B3.tif') 
                s2_b2_path = os.path.join(s2_chip_dir, 'B2.tif')
                
                s2_cloudmask_path = os.path.join(s2_chip_dir, 'LabelCloud.tif')

                required_files = [s1_vv_path, s1_vh_path, s2_b4_path, s2_b3_path, s2_b2_path, s2_cloudmask_path]
                if all(os.path.exists(p) for p in required_files):
                    records.append({
                        's1_vv': s1_vv_path,
                        's1_vh': s1_vh_path,
                        's2_b4_red': s2_b4_path,
                        's2_b3_green': s2_b3_path,
                        's2_b2_blue': s2_b2_path,
                        's2_cloudmask': s2_cloudmask_path,
                        'event': os.path.basename(event_dir),
                        's1_chip_id': os.path.basename(s1_chip_dir),
                        's2_chip_id': os.path.basename(s2_chip_dir)
                    })
                else:
                    logger.warning(f"Missing one or more files in chip pair: "
                                   f"S1: {os.path.basename(s1_chip_dir)}, S2: {os.path.basename(s2_chip_dir)}. Skipping.")

        if not records:
            logger.error("No valid records were created. This can happen if files are still missing or paths are incorrect.")
            return

        logger.info(f"Successfully processed {len(records)} complete chip pairs.")
        df = pd.DataFrame(records)

        logger.info(f"Creating train/validation split ({1-test_size:.0%}/{test_size:.0%})...")
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        final_df = pd.concat([train_df, val_df]).sort_values('event').reset_index(drop=True)
        
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Manifest file created successfully at: {output_csv}")
        logger.info(f"Dataset summary:\n{final_df['split'].value_counts()}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == '__main__':

    DATASET_ROOT = "data/raw/data/c2s_ms_floods/chips"
    
    OUTPUT_MANIFEST = "data/processed/data_manifest.csv"

    create_dataset_manifest(DATASET_ROOT, OUTPUT_MANIFEST)