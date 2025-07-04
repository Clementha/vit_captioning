# train.py
import torch
from tqdm import tqdm  
from torch.utils.data import DataLoader
from models.encoder import ViTEncoder
from models.decoder import TransformerDecoder
from utils import generate_square_subsequent_mask
from bert_score import score as bertscore 
import wandb
import sys

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

# We use only the vocab and tokenizer from a pretrained model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

from dataset import Flickr30kDataset
def trainFull():

    wandb.init(
        project="vit_captioning",  
        name="baseline-run",       
        config={
            "epochs": 40,
            "batch_size": 32,
            "max_length": 50,
            "learning_rate": 1e-4,
            "num_workers": 4,
            "model": "ViTEncoder + TransformerDecoder",
        }
    )

    BATCH_SIZE = wandb.config.batch_size
    NUM_EPOCHS = wandb.config.epochs
    NUM_WORKERS = wandb.config.num_workers
    MAX_LENGTH = wandb.config.max_length
    LEARNING_RATE = wandb.config.learning_rate
    ENCODER_LR = LEARNING_RATE * 0.1  # Example: lower LR for encoder
    UNFREEZE_EPOCH = int(NUM_EPOCHS * 0.3)  # Example: unfreeze encoder after 30% of epochs

    # ----------------------------
    # 2. Prepare dataset & dataloader
    # ----------------------------

    # Values defined in wandb is used throughout the training
    train_dataset = Flickr30kDataset(MAX_LENGTH)

    # Standard PyTorch DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    # Debug block goes here
    #sys.exit(0)

    # ----------------------------
    # 3. Initialize model
    # ----------------------------
    encoder = ViTEncoder().to(device)
    decoder = TransformerDecoder(vocab_size=vocab_size).to(device)

    # === Freeze ViT encoder ===
    for param in encoder.parameters():
        param.requires_grad = False

    print("Encoder frozen:", all(not p.requires_grad for p in encoder.parameters()))


    # ✅ Good sanity check
    print("Decoder training params:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    print("Encoder training params:", sum(p.numel() for p in encoder.parameters() if p.requires_grad))

    # Example: Combine encoder-decoder params
    #params = list(encoder.parameters()) + list(decoder.parameters())

    # ----------------------------
    # 4. Loss and optimizer
    # ----------------------------
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)  # adjust padding index if using tokenizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    # ----------------------------
    # 5. Training loop
    # ----------------------------
    
    for epoch in range(NUM_EPOCHS):

        if epoch == UNFREEZE_EPOCH:
            print(f"Unfreezing encoder at epoch {epoch+1}")
            for param in encoder.parameters():
                param.requires_grad = True
            # Recreate optimizer with both param groups
            optimizer = torch.optim.Adam([
                {"params": encoder.parameters(), "lr": ENCODER_LR},
                {"params": decoder.parameters(), "lr": LEARNING_RATE}
            ])

        encoder.train()
        decoder.train()

        # print("Encoder training mode:", encoder.training)
        # print("Decoder training mode:", decoder.training)

        # for name, param in encoder.named_parameters():
        #     print(f"{name} requires_grad: {param.requires_grad}")

        total_loss = 0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}",
            bar_format='{desc} {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}'
        )

        for batch_idx, (images, input_ids, attention_mask) in progress_bar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            encoder_outputs = encoder(images)  # [batch_size, encoder_dim]
            #encoder_outputs = encoder_outputs  # (keep as [batch_size, encoder_dim])

            outputs = decoder(input_ids, encoder_outputs, tgt_attention_mask=attention_mask)

            # Shift so inputs predict next token
            targets = input_ids[:, 1:]        # [batch_size, seq_len-1]
            outputs = outputs[:, :-1, :]      # [batch_size, seq_len-1, vocab_size]

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    "epoch": epoch+1,
                    "batch": batch_idx,
                    "loss": f"{loss.item():.4f}"
                })
            wandb.log({"batch_loss": loss.item()})

            if batch_idx % 200 == 0:
                logits = outputs[0]  # first sample in batch
                pred_ids = logits.argmax(dim=-1).tolist()
                generated_caption = tokenizer.decode(pred_ids, skip_special_tokens=True)

                target_ids = input_ids[0].tolist()
                target_caption = tokenizer.decode(target_ids, skip_special_tokens=True)

                print(f"\n[Batch {batch_idx}] Truth:     {target_caption}")
                print(f"[Batch {batch_idx}] Generated: {generated_caption}")

                    # ✅ Compute BERTScore
                P, R, F1 = bertscore(
                    [generated_caption],
                    [target_caption],
                    lang="en",
                    verbose=False
                )
                bert_f1 = F1[0].item()
                print(f"BERTScore F1: {bert_f1:.4f}")
                wandb.log({"bertscore_f1": bert_f1})


        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss})

    wandb.finish()

    # ----------------------------
    # 6. Save checkpoints (optional)
    # ----------------------------
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "./artifacts/vit_captioning.pth")

if __name__ == "__main__":
    trainFull()


