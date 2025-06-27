import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import logging
from logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

#1. Adversarial Loss
class LSGANLoss(nn.Module):
    '''
    Least Squares GAN Loss
    This loss function is used in LSGANs to stabilize training by minimizing the
    least squares error between the discriminator's predictions and the target labels.
    '''
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(LSGANLoss, self).__init__()
        # Use register_buffer to make these tensors part of the module's state,
        # but not model parameters. This ensures they are moved to the correct
        # device like GPU along with the module.
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        logger.debug("LSGANLoss called, target_is_real=%s", target_is_real)
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        # Expand the target label tensor to match the size of the prediction
        return self.loss(prediction, target_tensor.expand_as(prediction))

#2. Perceptual Loss - VGG19 Feature Loss
class PerceptualLoss(nn.Module):
    '''
    Calculates perceptual loss using a pre-trained VGG19 network.
    This loss compares intermediate features of generated and real images,
    focusing on texture and structural similarity rather than just pixel differences.
    '''
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.L1Loss()

        #ImageNet Normalization values
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) #ImageNet mean
        # ImageNet std deviation
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen_img, real_img):
        logger.debug("PerceptualLoss forward with gen_img shape %s", tuple(gen_img.shape))
        # VGG expects 3-channel input normalized with ImageNet stats
        # Our images are in [-1, 1], so we scale to [0, 1] first
        gen_img = (gen_img + 1) / 2
        real_img = (real_img + 1) / 2

        gen_img = (gen_img - self.mean) / self.std
        real_img = (real_img - self.mean) / self.std


        gen_features = self.vgg(gen_img)
        real_features = self.vgg(real_img)

        return self.loss(gen_features, real_features)
    
#Custom Speckle Preservation Loss
class SpecklePreservationLoss(nn.Module):
    '''
    A custom edge-aware loss to encourage the generator to maintain smooth areas from the SAR image,
    preventing it from hallucinating high-frequency details where none should exist.
    '''
    def __init__(self):
        super(SpecklePreservationLoss, self).__init__()

        #Sobel filters to compute image gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) #Horizontal Sobel filter
        #Vertical Sobel filter
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.loss = nn.L1Loss()

    def get_gradient_magnitude(self, img):
        if img.shape[1] == 3:
            # Convert to grayscale using standard luminance weights
            img_gray = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        elif img.shape[1] == 2:
            img_gray = img[:, 0:1, :, :]
        else:
            img_gray =img

        grad_x = F.conv2d(img_gray, self.sobel_x, padding='same')
        grad_y = F.conv2d(img_gray, self.sobel_y, padding='same')
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, gen_img, sar_img):
        logger.debug("SpecklePreservationLoss forward")
        grad_gen = self.get_gradient_magnitude(gen_img)
        grad_sar = self.get_gradient_magnitude(sar_img)

        # We want to penalize gradients in the generated image where the SAR is smooth.
        # Create a weight map: high weight where SAR gradient is low, and low weight where it's high.
        weight_map = torch.exp(-grad_sar)

        # This loss encourages the weighted generator gradient to be close to zero.
        loss = self.loss(weight_map * grad_gen, torch.zeros_like(grad_gen))
        return loss

class WaterIndexConsistencyLoss(nn.Module):
    '''
    Enforces that areas known to be water in the SAR image are mapped to pixels
    that have a high water index (NDWI) in the generated optical image.
    '''
    def __init__(self, sar_water_threshold_db=-22.0):
        super(WaterIndexConsistencyLoss, self).__init__()
        self.sar_water_threshold_db = sar_water_threshold_db
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, gen_img, sar_img):
        logger.debug("WaterIndexConsistencyLoss forward")
        # Step 1: Create a "weak" water mask from the SAR VH channel (channel 1)
        # Denormalize SAR from [-1, 1] -> [0, 1] -> [-25, 0] dB range
        sar_vh_db = ((sar_img[:, 1:2, :, :] + 1) / 2) * 25 - 25
        sar_water_mask = (sar_vh_db < self.sar_water_threshold_db).float()

        # Step 2: Calculate a proxy for NDWI on the generated optical image
        # NDWI = (Green - NIR) / (Green + NIR). We use Red as a proxy for NIR.
        gen_img_01 = (gen_img + 1) / 2  # Scale to [0, 1]
        green = gen_img_01[:, 1:2, :, :]  # Green channel
        red = gen_img_01[:, 0:1, :, :]  # Red channel

        ndwi_proxy = (green - red) / (green + red + 1e-6)  # Avoid division by zero

        # The output of NDWI is in [-1, 1], which can be treated as logits.
        # I used BCEWithLogitsLoss to compare these logits with the SAR-derived water mask.
        # This pushes the model to generate high NDWI values (water) where the SAR detects water.
        return self.loss(ndwi_proxy, sar_water_mask)