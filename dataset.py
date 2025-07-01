# dataset.py

from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor

class CaptionDataset(Dataset):
    def __init__(self, samples, tokenizer):
        """
        samples: list of (image_path, caption)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        # ✅ This line sets up the image -> tensor preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        # ✅ Convert PIL image to tensor
        image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        # ✅ Tokenize caption
        #input_ids = self.tokenizer.encode(caption, return_tensors="pt").squeeze(0)
        input_ids = self.tokenizer.encode(
            caption,
            add_special_tokens=True,   # ✅ adds [CLS] and [SEP]
            return_tensors="pt"
        ).squeeze(0)

        return image_tensor, input_ids