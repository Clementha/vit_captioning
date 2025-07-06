# dataset.py

from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor, CLIPProcessor

from datasets import load_dataset
#from torchvision import transforms
from transformers import AutoTokenizer
import torch
from wandb import config

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, model="None", flatten_captions=False, max_token_length=50):

        # Debug
        print(f"Clement model: {model}")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_token_length = max_token_length
        # Load from cache if exists, else download
        self.dataset = load_dataset("nlphuji/flickr30k", split="test")
        self.flatten_captions = flatten_captions
        self.captions_per_image = 5  # Number of captions per image in Flickr30k

        if model == "CLIPEncoder":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif model == "ViTEncoder":
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            raise ValueError("Unknown model type. Use 'CLIPEncoder' or 'ViTEncoder'.")
          

    def __len__(self):
        if self.flatten_captions:
            return len(self.dataset) * self.captions_per_image
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        if self.flatten_captions:
            # Calculate the actual index in the dataset
            actual_idx = idx // self.captions_per_image
            caption_idx = idx % self.captions_per_image
        else:
            actual_idx = idx
            caption_idx = 0 # Default to the first caption if not flattening            

        example = self.dataset[actual_idx]

        # Load image
        image = example['image'].convert('RGB')

        encoding = self.processor(images=image, return_tensors='pt')
        pixel_values = encoding['pixel_values'].squeeze(0)

        # Tokenize caption
        caption = example['caption']
        if isinstance(caption, list):   # If there are multiple captions, choose one
            caption = caption[caption_idx]  # or random.choice(caption)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_length,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            return_tensors="pt"
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return pixel_values, input_ids, attention_mask
    
    # def save_to_disk(self, path="data/my_flickr30k"):
    #     self.dataset.save_to_disk(path)


# if __name__ == "__main__":
#         ds = Flickr30kDataset()
#         ds.save_to_disk("data/my_flickr30k")
#         print("Saved Flickr30k to data/my_flickr30k")
