from transformers import AutoTokenizer
from models.decoder import TinyTransformerDecoder
from models.encoder import ViTEncoder
from utils import generate_square_subsequent_mask
import torch
import torch.nn.functional as F
from PIL import Image

# === 1. Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# === 2. Load decoder ===
decoder = TinyTransformerDecoder(vocab_size=tokenizer.vocab_size)
decoder.load_state_dict(torch.load("./artifacts/decoder.pth", map_location='cpu'))
decoder.eval()

# === 3. Load encoder ===
encoder = ViTEncoder()

# === 4. Process test image ===
image = Image.open("./images/manOnBike.png").convert("RGB")
image_tensor = encoder.processor(images=image, return_tensors="pt")['pixel_values']
memory = encoder.encode(image_tensor)

# === 5. Generate ===
from utils import generate_square_subsequent_mask

def generate(decoder, memory, tokenizer, max_len=20):
    device = 'cpu'
    decoder.eval()
    memory = memory.to(device)
    tgt_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

    for step in range(max_len):
        tgt_seq_len = tgt_input_ids.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)

        logits = decoder(tgt_input_ids, memory, tgt_mask=tgt_mask)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tgt_input_ids = torch.cat([tgt_input_ids, next_token], dim=1)

        # if tokenizer.sep_token_id and next_token.item() == tokenizer.sep_token_id:
        #     break
        if tokenizer.sep_token_id is not None and next_token.item() == tokenizer.sep_token_id:
            break

    return tgt_input_ids

generated_ids = generate(decoder, memory, tokenizer)
print("Generated IDs:", generated_ids)

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text:", text)