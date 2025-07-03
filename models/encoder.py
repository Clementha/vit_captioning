# models/encoder.py
from transformers import ViTModel, ViTImageProcessor
import torch.nn as nn
import torch
from PIL import Image


import torch.nn as nn

class ViTEncoder(nn.Module):
    def __init__(self):  # Make decoder_dim configurable!
        super(ViTEncoder, self).__init__()

        #weights = ViT_B_16_Weights.DEFAULT
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding