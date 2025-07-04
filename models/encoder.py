# models/encoder.py
from transformers import ViTModel, ViTImageProcessor, CLIPModel
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

        # ViTModel - output shape = [batch, seq_len, hidden]
        outputs = self.vit(pixel_values=pixel_values)

        # Take CLS: last_hidden_state

        cls_embedding = outputs.last_hidden_state[:, 0]
        return cls_embedding
    
# encoder.py
from transformers import CLIPModel

class CLIPEncoder(nn.Module):
    def __init__(self):
        super(CLIPEncoder, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, pixel_values):
        # ✅ Directly get the pooled image features (already the final representation)
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return image_features  # shape: [batch_size, hidden_dim]