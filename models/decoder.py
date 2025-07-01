# models/decoder.py
import torch
import torch.nn as nn

class TinyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TinyTransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, d_model=768, nhead=8, dim_feedforward=2048, vocab_size=30522, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([
            TinyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input_ids, memory, tgt_mask=None, memory_mask=None):
        tgt_emb = self.embedding(tgt_input_ids) * (self.d_model ** 0.5)
        seq_len = tgt_emb.size(1)
        tgt_emb = tgt_emb + self.pos_embedding[:, :seq_len, :]
        tgt = tgt_emb.transpose(0, 1)
        memory = memory.transpose(0, 1)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        tgt = self.norm(tgt)
        tgt = tgt.transpose(0, 1)
        logits = self.output_head(tgt)
        return logits