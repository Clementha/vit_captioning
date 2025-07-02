# models/encoder.py
from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
import torch
from PIL import Image

from torchvision.models import vit_b_16, ViT_B_16_Weights   
class ViTEncoder(nn.Module):
    def __init__(self):
        super(ViTEncoder, self).__init__()

        # Load the pre-trained ViT backbone
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)

        # Remove the classification head so it outputs features
        self.vit.heads = nn.Identity()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3, H, W)
        returns: features tensor
        """
        features = self.vit(x)  # shape: (batch_size, hidden_dim)
        return features