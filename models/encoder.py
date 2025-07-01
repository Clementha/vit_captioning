# models/encoder.py
from transformers import ViTModel, ViTImageProcessor
import torch

class ViTEncoder:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        from PIL import Image
        import torch

        self.model = ViTModel.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, image_tensor):
        """
        image_tensor: shape [batch_size, 3, H, W]
        """
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor)
        return outputs.last_hidden_state