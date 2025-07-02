# models/decoder.py
import torch
import torch.nn as nn

class decoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Multi-head self-attention (masked)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Multi-head cross-attention (encoder-decoder attention)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms for residual connections   
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):

        # Masked Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # --- Cross-Attention (Encoder-Decoder Attention) ---
        # Query: decoder output so far; Key & Value: encoder output
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# === Full Transformer Decoder ===
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=768, nhead=8, dim_feedforward=2048, vocab_size=30522, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([
            decoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Final projection to vocab
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input_ids, memory, tgt_mask=None, memory_mask=None):
        tgt_emb = self.embedding(tgt_input_ids) * (self.d_model ** 0.5)
        seq_len = tgt_emb.size(1)
        tgt_emb = tgt_emb + self.pos_embedding[:, :seq_len, :]

        # Transformer expects shape: (tgt_len, batch_size, d_model)
        tgt = tgt_emb.transpose(0, 1)
        #memory = memory.transpose(0, 1)

        # Pass through each DecoderLayer
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        tgt = self.norm(tgt)

        # Final projection to vocabulary logits
        tgt = tgt.transpose(0, 1)
        logits = self.output_head(tgt)
        return logits
