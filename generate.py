# generate.py

import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer
from models.encoder import ViTEncoder
from models.decoder import TransformerDecoder  # your Decoder must have a generate() method


# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------------------------------
# 1️⃣ Load encoder, decoder, tokenizer
# --------------------------------------------
encoder = ViTEncoder().to(device)
decoder = TransformerDecoder(
    vocab_size=30522,   # For example, BERT vocab size if you use bert-base-uncased
    hidden_dim=512      # Match your model's hidden_dim
).to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load checkpoint
checkpoint = torch.load('./artifacts/vit_captioning.pth', map_location=device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

# --------------------------------------------
# 2️⃣ Setup image processor
# --------------------------------------------
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# --------------------------------------------
# 3️⃣ Load and preprocess your image
# --------------------------------------------
image = Image.open("./images/debug_image.jpg").convert("RGB")
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