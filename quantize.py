"""
quantize.py

Usage:
  python quantize.py --input artifacts/your_checkpoint.pth --output artifacts/your_checkpoint_quantized.pth --encoder ViTEncoder

This script loads your existing .pth checkpoint, applies dynamic quantization
to the TransformerDecoder (and optionally your encoder), and saves a new .pth
file that's smaller and runs faster on CPU.

✅ Make sure your models/encoder.py and models/decoder.py are importable!
"""

import argparse
import torch
from models.decoder import TransformerDecoder
from models.encoder import ViTEncoder, CLIPEncoder

def main():
    parser = argparse.ArgumentParser(description="Quantize your checkpoint for smaller size & faster inference.")
    parser.add_argument("--input", required=True, help="Path to original .pth checkpoint")
    parser.add_argument("--output", required=True, help="Path to save quantized .pth checkpoint")
    parser.add_argument("--encoder", default="ViTEncoder", choices=["ViTEncoder", "CLIPEncoder"], help="Encoder type")
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.backends.quantized.engine = "fbgemm"

    # Load your original checkpoint
    checkpoint = torch.load(args.input, map_location=device)

    # Instantiate encoder
    if args.encoder == "ViTEncoder":
        encoder = ViTEncoder()
    elif args.encoder == "CLIPEncoder":
        encoder = CLIPEncoder()
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    # Instantiate decoder
    encoder_dim = 768 if args.encoder == "ViTEncoder" else 512
    decoder = TransformerDecoder(
        vocab_size=30522,   # or match your tokenizer vocab size
        hidden_dim=encoder_dim,
        encoder_dim=encoder_dim
    )
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    decoder.eval()

    # ✅ Apply dynamic quantization
    print("Quantizing TransformerDecoder...")
    decoder_quantized = torch.quantization.quantize_dynamic(
        decoder, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Optional: quantize encoder too if it's all torch layers
    # If you want to quantize your encoder, you can try:
    print("Quantizing encoder...")
    encoder_quantized = torch.quantization.quantize_dynamic(
        encoder, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save new checkpoint
    torch.save({
        'encoder_state_dict': encoder_quantized.state_dict(),
        'decoder_state_dict': decoder_quantized.state_dict(),
    }, args.output)

    print(f"✅ Quantized checkpoint saved to: {args.output}")

if __name__ == "__main__":
    main()