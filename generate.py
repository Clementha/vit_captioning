# generate.py

import torch
from PIL import Image
from transformers import ViTImageProcessor, CLIPProcessor, AutoTokenizer
from models.encoder import ViTEncoder, CLIPEncoder
from models.decoder import TransformerDecoder  # your Decoder must have a generate() method
import argparse # Import the argparse module

# Setup device
# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU acceleration.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU acceleration.")
else:
    device = torch.device("cpu")
    print("No GPU found, falling back to CPU.")

# --------------------------------------------
# 0️⃣ Parse command line arguments
# --------------------------------------------
parser = argparse.ArgumentParser(description="Generate caption using ViT or CLIP.")
parser.add_argument("--model", type=str, default="ViTEncoder",
                    choices=["ViTEncoder", "CLIPEncoder"],
                    help="Choose encoder: ViTEncoder or CLIPEncoder")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the .pth checkpoint file")
parser.add_argument("--image", type=str, required=True,
                    help="Path to input image file")
args = parser.parse_args()

# --------------------------------------------
# 1️⃣ Load encoder, decoder, tokenizer
# --------------------------------------------

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# --------------------------------------------
# 2️⃣ Setup image processor
# --------------------------------------------
# Select encoder, processor, output dim
if args.model == "ViTEncoder":
    encoder = ViTEncoder().to(device)
    encoder_dim = 768
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
elif args.model == "CLIPEncoder":
    encoder = CLIPEncoder().to(device)
    encoder_dim = 512
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
else:
    raise ValueError("Unknown model type")

decoder = TransformerDecoder(
    vocab_size=30522,   # For example, BERT vocab size if you use bert-base-uncased
    hidden_dim=encoder_dim,
    encoder_dim=encoder_dim      # Match your model's hidden_dim
).to(device)


# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()


# --------------------------------------------
# 3️⃣ Load and preprocess your image
# --------------------------------------------
image = Image.open(args.image).convert("RGB")
encoding = processor(images=image, return_tensors='pt')
pixel_values = encoding['pixel_values'].to(device)

# --------------------------------------------
# 4️⃣ Pass through encoder -> decoder.generate()
# --------------------------------------------
with torch.no_grad():
    encoder_outputs = encoder(pixel_values)
    generated_ids = decoder.generate(
        encoder_outputs,
        tokenizer,
        max_length=32,            # Change as needed
        sos_token_id=tokenizer.cls_token_id,  # [CLS] token for BERT
        eos_token_id=tokenizer.sep_token_id   # [SEP] token for BERT
    )

# --------------------------------------------
# 5️⃣ Decode tokens to text
# --------------------------------------------
generated_caption = decoder.generate(
    encoder_outputs,
    tokenizer,
    max_length=32,
    sos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id
)
print(f"Generated caption: {generated_caption}")