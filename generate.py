import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.encoder import ViTEncoder
from models.decoder import TransformerDecoder

# === 1. Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === 3. Load encoder and decoder ===
encoder = ViTEncoder().to(device)
encoder.load_state_dict(torch.load("encoder.pth"))
encoder.eval()

decoder = TransformerDecoder(
    num_layers=6,
    d_model=768,
    nhead=8,
    dim_feedforward=2048,
    vocab_size=tokenizer.vocab_size
).to(device)
decoder.load_state_dict(torch.load("decoder.pth"))
decoder.eval()

# === 4. Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# === 5. Load image ===
image = Image.open("test_image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

# === 6. Encode ===
with torch.no_grad():
    features = encoder(image_tensor)  # [batch_size, d_model]
    features = features.unsqueeze(0)  # [1, batch_size, d_model]

# === 7. Greedy decode ===
max_len = 50
generated = [tokenizer.cls_token_id]  # e.g., use [CLS] as BOS

for _ in range(max_len):
    input_ids = torch.tensor(generated).unsqueeze(0).to(device)  # [1, seq_len]

    with torch.no_grad():
        outputs = decoder(input_ids, features)  # [1, seq_len, vocab_size]

    next_token_logits = outputs[0, -1, :]  # last token logits
    next_token_id = torch.argmax(next_token_logits).item()

    if next_token_id == tokenizer.sep_token_id:
        break

    generated.append(next_token_id)

caption = tokenizer.decode(generated, skip_special_tokens=True)
print("Generated Caption:", caption)