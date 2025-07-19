# generate.py

import torch
from PIL import Image
from transformers import ViTImageProcessor, CLIPProcessor, AutoTokenizer
from models.encoder import ViTEncoder, CLIPEncoder
from models.decoder import TransformerDecoder

import argparse


class CaptionGenerator:
    def __init__(self, model_type: str, checkpoint_path: str, quantized=False):
        print(f"Loading {model_type} | Quantized: {quantized}")
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA CUDA GPU acceleration.")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS GPU acceleration.")
        else:
            self.device = torch.device("cpu")
            print("No GPU found, falling back to CPU.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Select encoder, processor, output dim
        if model_type == "ViTEncoder":
            self.encoder = ViTEncoder().to(self.device)
            self.encoder_dim = 768
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        elif model_type == "CLIPEncoder":
            self.encoder = CLIPEncoder().to(self.device)
            self.encoder_dim = 512
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError("Unknown model type")

        if quantized:
            print("Applying dynamic quantization to encoder...")
            self.encoder = torch.ao.quantization.quantize_dynamic(
                self.encoder,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        # Initialize decoder
        self.decoder = TransformerDecoder(
            vocab_size=30522,
            hidden_dim=self.encoder_dim,
            encoder_dim=self.encoder_dim
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder.eval()
        self.decoder.eval()

    def generate_caption(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(images=image, return_tensors='pt')
        pixel_values = encoding['pixel_values'].to(self.device)

        captions = {}

        with torch.no_grad():
            encoder_outputs = self.encoder(pixel_values)

            # Greedy
            caption_ids = self.decoder.generate(encoder_outputs, mode="greedy")
            captions['greedy'] = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)

            # Top-k
            caption_ids = self.decoder.generate(encoder_outputs, mode="topk", top_k=30)
            captions['topk'] = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)

            # Top-p
            caption_ids = self.decoder.generate(encoder_outputs, mode="topp", top_p=0.92)
            captions['topp'] = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)

        return captions


if __name__ == "__main__":
    # CLI usage
    parser = argparse.ArgumentParser(description="Generate caption using ViT or CLIP.")
    parser.add_argument("--model", type=str, default="ViTEncoder",
                        choices=["ViTEncoder", "CLIPEncoder"],
                        help="Choose encoder: ViTEncoder or CLIPEncoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image file")
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Load encoder with dynamic quantization"
    )  ### ✅ ADDED

    args = parser.parse_args()

    generator = CaptionGenerator(
        model_type=args.model,
        checkpoint_path=args.checkpoint
    )

    captions = generator.generate_caption(args.image)

    print(f"Greedy-argmax (deterministic, factual): {captions['greedy']}")
    print(f"Top-k (diverse, creative): {captions['topk']}")
    print(f"Top-p (diverse, human-like): {captions['topp']}")