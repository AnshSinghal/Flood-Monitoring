import torch
import torch.utils.data as DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandBLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser

from lightning_module import SAR2OpticalGAN
from dataset import FloodDataset

def main(hparams):
    #Initialize Model
    model = SAR2OpticalGAN(hparams)

    #initialize dataset and loaders
    train_dataset = FloodDataset(manifest_path=hparams.manifest_path, split='train')
    val_dataset = FloodDataset(manifest_path=hparams.manifest_path, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True
    )