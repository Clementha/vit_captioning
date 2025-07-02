# train.py
import torch
from tqdm import tqdm  
from torch.utils.data import DataLoader
from models.encoder import ViTEncoder
from models.decoder import TransformerDecoder
from utils import generate_square_subsequent_mask
import wandb



# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# We use only the vocab and tokenizer from a pretrained model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

from dataset import Flickr30kDataset
def trainFull():
    # ----------------------------
    # 2. Prepare dataset & dataloader
    # ----------------------------
    train_dataset = Flickr30kDataset(split="train")

    # Optional: save to disk once if you want
    # train_dataset.save_to_disk("data/my_flickr30k")

    # Standard PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    wandb.init(
        project="vit_captioning",  # Replace with your project name
        name="baseline-run",       # Optional: unique run name
        config={
            "epochs": 10,
            "batch_size": 32,
            "max_length": 50,
            "learning_rate": 1e-4,
            "model": "ViTEncoder + TransformerDecoder",
        }
    )

    # ----------------------------
    # 3. Initialize model
    # ----------------------------
    encoder = ViTEncoder().to(device)
    decoder = TransformerDecoder(vocab_size=vocab_size).to(device)

    # Example: Combine encoder-decoder params
    params = list(encoder.parameters()) + list(decoder.parameters())

    # ----------------------------
    # 4. Loss and optimizer
    # ----------------------------
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # adjust padding index if using tokenizer
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # ----------------------------
    # 5. Training loop
    # ----------------------------
    num_epochs = 10
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (images, input_ids) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            input_ids = input_ids.to(device)

            # Forward pass
            features = encoder(images)
            # print("features shape:", features.shape)
            # print("pos_embedding shape:", decoder.pos_embedding.shape)
            features = features.unsqueeze(0)  #Now: [1, batch_size, d_model]

            outputs = decoder(input_ids, features)

            # Shift so inputs predict next token
            targets = input_ids[:, 1:]        # [batch_size, seq_len-1]
            outputs = outputs[:, :-1, :]      # [batch_size, seq_len-1, vocab_size]

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))


            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # ✅ Keep progress tidy
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    "epoch": epoch+1,
                    "batch": batch_idx,
                    "loss": f"{loss.item():.4f}"
                })

            wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss})

    wandb.finish()

    # ----------------------------
    # 6. Save checkpoints (optional)
    # ----------------------------
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "artifacts/vit_captioning.pth")

if __name__ == "__main__":
    trainFull()
