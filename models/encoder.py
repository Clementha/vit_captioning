# models/encoder.py
from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
import torch
from PIL import Image

from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self, decoder_dim=768):  # Make decoder_dim configurable!
        super(ViTEncoder, self).__init__()

        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)

        # Remove the classification head so it outputs features
        self.vit.heads = nn.Identity()

        vit_out_dim = 768  # ViT-B/16
        # Projection layer: can be identity if dims match
        if vit_out_dim != decoder_dim:
            self.proj = nn.Linear(vit_out_dim, decoder_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        features = self.vit(x)  # (batch_size, vit_out_dim)
        #print("Encoder raw ViT features shape:", features.shape)
        projected = self.proj(features)  # (batch_size, decoder_dim)
        #print("Encoder projected features shape:", projected.shape)
        return projected