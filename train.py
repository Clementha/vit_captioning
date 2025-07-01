# train.py
import torch
from torch.utils.data import DataLoader
from models.encoder import ViTEncoder
from models.decoder import TinyTransformerDecoder
from utils import generate_square_subsequent_mask
from dataset import CaptionDataset

def train():
    # Dummy tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # Dummy dataset
    samples = [
        ("./images/girl.png", "A little girl climbing into a wooden playhouse."),
        ("./images/dog.png", "A dog running in the yard."),
        ("./images/manOnBike.png", "A man riding a bike."),
        ("./images/Picnic.png", "A family having a picnic.")
    ]
    dataset = CaptionDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    encoder = ViTEncoder()
    decoder = TinyTransformerDecoder(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder.to(device)

    for epoch in range(20):
        for image_tensor, tgt_input_ids in dataloader:
            memory = encoder.encode(image_tensor)  # ✅ use tensor here!
            memory = memory.to(device)
            tgt_input_ids = tgt_input_ids.to(device)

            tgt_in = tgt_input_ids[:, :-1]
            tgt_out = tgt_input_ids[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1)).to(device)

            logits = decoder(tgt_in, memory, tgt_mask=tgt_mask)
            logits = logits.view(-1, logits.size(-1))
            tgt_out = tgt_out.view(-1)

            loss = criterion(logits, tgt_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
    
    torch.save(decoder.state_dict(), "./artifacts/decoder.pth")
    print("✅ Saved trained decoder to decoder.pth")

if __name__ == "__main__":
    train()