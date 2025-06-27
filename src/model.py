import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()

        #using pre-trained resnet34 as encoder
        self.base_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        #modifying the first layer to accept 2 channels (VV and VH)\
        self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        #encoder layers from resnet34
        self.encoder1 = nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu)
        self.encoder2 = self.base_model.layer1
        self.encoder3 = self.base_model.layer2
        self.encoder4 = self.base_model.layer3
        self.encoder5 = self.base_model.layer4

        #decoder layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1),
            nn.Tanh()  # Output layer with Tanh activation for pixel values in range [-1, 1]
        )

    def forward(self, x):
        logger.debug("UNetGenerator forward with input shape %s", tuple(x.shape))
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        d4 = self.upconv4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        # Note: ResNet conv1 output has different size, so we upsample e1
        e1_upsampled = nn.functional.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1_upsampled], dim=1)
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)
    

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, sar_img, opt_img):
        logger.debug("PatchGANDiscriminator forward with SAR shape %s and OPT shape %s", tuple(sar_img.shape), tuple(opt_img.shape))
        img_input = torch.cat([sar_img, opt_img], dim=1)  # Concatenate SAR and optical images
        return self.model(img_input)  # Forward pass through the discriminator