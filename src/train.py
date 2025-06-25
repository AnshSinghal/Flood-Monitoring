import torch
import yaml
from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandBLogger

from lightning_module import SAR2OpticalGAN
from datamodule import SARDataModule

from model import UNetGenerator, PatchGANDiscriminator
from losses import PerceptualLoss, LSGANLoss, SpecklePreservationLoss, WaterIndexConsistencyLoss
import pytorch_lightning as pl

def main(hparams):
    wandb_logger = WandBLogger(
        project=hparams.project_name,
        name=hparams.run_name,
    )

    wandb_logger.log_hyperparams(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{hparams.run_name}",
        filename='{epoch:02d}-{val_psnr:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val/psnr', # Monitor a validation metric like PSNR
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = SAR2OpticalGAN(hparams)
    datamodule = SARDataModule(hparams)

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=hparams.accelerator,
        devices=1,
        precision=hparams.precision,
        log_every_n_steps=hparams.log_every_n_steps,
    )
    trainer.fit(model, datamodule)

    wandb_logger.experiment.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hparams = Namespace(**config)
    main(hparams)