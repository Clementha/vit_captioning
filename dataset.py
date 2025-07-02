# dataset.py

from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor

from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer
import torch

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, split="test"):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 50  # Adjust as needed
        # Load from cache if exists, else download
        self.dataset = load_dataset("nlphuji/flickr30k", split="test")

        # Define your image transforms (ViT/CLIP style)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # adjust for your model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # example
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Transform image
        image = item['image']
        image = self.transform(image)

        # Grab caption
        caption = item['caption']  # Should be just a string
        if isinstance(caption, list):
            caption = caption[0]  # or random.choice(caption)
        # Tokenize: convert to token IDs
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # print("Caption:", caption)
        # print("tokens.input_ids.shape:", tokens.input_ids.shape)

        input_ids = tokens.input_ids.squeeze(0)  # (max_length,)
        # Always ensure tensor is long!
        input_ids = input_ids.long()
        #print("Input IDs dtype:", input_ids.dtype)

        return image, input_ids

    def save_to_disk(self, path="data/my_flickr30k"):
        self.dataset.save_to_disk(path)


if __name__ == "__main__":
        ds = Flickr30kDataset()
        ds.save_to_disk("data/my_flickr30k")
        print("Saved Flickr30k to data/my_flickr30k")
