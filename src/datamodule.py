from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import FloodDataset

def collate_fn(batch):
    '''
    Custom collate function to filter out None values.
    useful if __getitem__ can return None on an error.
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch) if batch else (None, None)

class SARDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        '''
        called on every GPU/TPU in distributed training
        '''
        if stage == 'fit' or stage is None:
            self.train_dataset = FloodDataset(
                manifest_path=self.hparams.manifest_path,
                split='train',
                cloud_threshold=self.hparams.cloud_threshold,
                use_speckle_filter=self.hparams.use_speckle_filter
            )
            self.val_dataset = FloodDataset(
                manifest_path=self.hparams.manifest_path,
                split='val',
                cloud_threshold=100.0, # No cloud threshold for validation, we want to see performance on all data
                use_speckle_filter=self.hparams.use_speckle_filter
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate_fn
        )