import torch
import pytorch_lightning as pl
from model import UNetGenerator, PatchGANDiscriminator
from losses import LSGANLoss, PerceptualLoss, SpecklePreservationLoss, WaterIndexConsistencyLoss

class SAR2OpticalGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.generator = UNetGenerator(in_channels=2, out_channels=3)
        self.discriminator = PatchGANDiscriminator(in_channels=5)

        self.adv_loss = LSGANLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.speckle_loss = SpecklePreservationLoss()
        self.water_loss = WaterIndexConsistencyLoss()

        self.validation_z = None  # Placeholder for validation input

    def forward(self, sar_img):
        '''
        Forward pass through the generator.
        '''
        return self.generator(sar_img)
    
    def training_step(self, batch, batch_size, optimizer_idx):
        sar_img, opt_img = batch

        if optimizer_idx == 0:
            generated_opt = self(sar_img)

            pred_fake = self.discriminator(sar_img, generated_opt)
            loss_g_adv = self.adv_loss(pred_fake, target_is_real=True) #Fool the discriminator

            #Reconstruction and Physics Aware loss
            loss_g_l1 = self.l1_loss(generated_opt, opt_img)
            loss_g_perc = self.perceptual_loss(generated_opt, opt_img)
            loss_g_speckle = self.speckle_loss(generated_opt, sar_img)
            loss_g_water = self.water_loss(generated_opt, sar_img)

            #Total generator loss
            loss_g = (
                loss_g_adv * self.hparams.lambda_adv +
                loss_g_l1 * self.hparams.lambda_l1 +
                loss_g_perc * self.hparams.lambda_perc +
                loss_g_speckle * self.hparams.lambda_speckle +
                loss_g_water * self.hparams.lambda_water
            )

            self.log_dict({
                'loss_g': loss_g,
                'loss_g_adv': loss_g_adv,
                'loss_g_l1': loss_g_l1,
                'loss_g_perc': loss_g_perc,
                'loss_g_speckle': loss_g_speckle,
                'loss_g_water': loss_g_water
            })
            return loss_g
        
        if optimizer_idx == 1:
            generated_opt = self(sar_img).detach() # Detach to avoid backprop to generator

            #real image loss
            pred_real = self.discriminator(sar_img, opt_img)
            loss_d_real = self.adv_loss(pred_real, target_is_real=True)

            #fake image loss
            pred_fake = self.discriminator(sar_img, generated_opt)
            loss_d_fake = self.adv_loss(pred_fake, target_is_real=False)

            #Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5

            self.log_dict({
                'loss_d': loss_d,
                'loss_d_real': loss_d_real,
                'loss_d_fake': loss_d_fake
            }, prog_bar=True)
            return loss_d
        
    def visualization_step(self, batch, batch_idx):
        sar_img, opt_img = batch
        generated_opt = self(sar_img)

        val_l1_loss = self.l1_loss(generated_opt, opt_img)
        self.log('val_l1_loss', val_l1_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.validation_z is None:
            self.validation_z = batch

    def on_validation_epoch_end(self):
       #log images to weights and biases
        if self.logger and self.validation_z is not None:
           sar_img, opt_img = self.validation_z
           generated_opt = self(sar_img.to(self.device))

           #Denormalize images from [-1, 1] to [0, 1] for logging
           sar_img_01 = (sar_img[:, 0:1] + 1) / 2 #Just show the VV
           opt_img_01 = (opt_img + 1) / 2
           generated_opt_01 = (generated_opt + 1) / 2


           grid = torch.cat([sar_img_01, generated_opt_01, opt_img_01], dim=-1)
           self.logger.experiment.log({
                f"val/generated_images": [wandb.Image(img) for img in grid]
            })
        self.validation_z = None # Reset for next epoch

    def configure_optimizers(self):
        '''
        Configure optimizers and learning rate schedulers.
        '''
        lr = self.hparams.lr # Learning rate from hyperparameters
        b1 = self.hparams.b1 # Beta1 from hyperparameters
        b2 = self.hparams.b2 # Beta2 from hyperparameters

        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, betas=(b1, b2)) # AdamW optimizer for generator
        # AdamW optimizer for discriminator
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        # Learning rate scheduler
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.hparams.epochs) # Cosine annealing scheduler for generator
        # Cosine annealing scheduler for discriminator
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.hparams.epochs)


        return [opt_g, opt_d], [scheduler_g, scheduler_d]