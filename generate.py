# generate.py

import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from models.encoder import ViTEncoder
from models.decoder import TransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. Load tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# -------------------------------
# 2. Image transform (must match training!)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# -------------------------------
# 3. Load encoder (frozen) and decoder
# -------------------------------
encoder = ViTEncoder().to(device)
decoder = TransformerDecoder(vocab_size=tokenizer.vocab_size).to(device)

# Make sure encoder is frozen
for param in encoder.parameters():
    param.requires_grad = False

encoder.eval()
decoder.eval()

# -------------------------------
# 4. Load trained weights
# -------------------------------
checkpoint = torch.load("./artifacts/vit_captioning.pth", map_location=device)
decoder.load_state_dict(checkpoint['decoder_state_dict'])
print("Loaded decoder weights.")

# -------------------------------
# 5. Load and preprocess test image
# -------------------------------
image = Image.open("./images/Picnic.png").convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

# -------------------------------
# 6. Encode image → get features
# -------------------------------
with torch.no_grad():
    features = encoder(image)
    print("Encoder features mean:", features.mean().item(), "std:", features.std().item())
    print("Encoder features shape:", features.shape)  # (1, 768)

    memory = features.unsqueeze(0)  # (1, batch, d_model)
    print("Memory shape:", memory.shape)  # should be (1, 1, d_model)
    #print("Encoder features mean:", features.mean().item(), "std:", features.std().item())



# -------------------------------
# 7. Greedy decode with top-k sampling
# -------------------------------
max_length = 50
top_k = 5

generated_ids = [tokenizer.cls_token_id]  # e.g., [101]

for _ in range(max_length):
    input_ids = torch.tensor([generated_ids], device=device)  # (1, seq_len)

    with torch.no_grad():
        logits = decoder(input_ids, memory)  # (1, seq_len, vocab_size)
    
    next_token_logits = logits[0, -1, :]  # (vocab_size,)

    # Top-k sampling
    topk_probs = torch.topk(next_token_logits, k=top_k)
    next_token_id = topk_probs.indices[torch.randint(0, top_k, (1,))].item()

    # Stop if EOS or SEP token
    if next_token_id == tokenizer.sep_token_id:
        break

    generated_ids.append(next_token_id)

print("Generated IDs:", generated_ids)

# -------------------------------
# 8. Decode IDs to caption
# -------------------------------
caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Generated Caption:", caption)