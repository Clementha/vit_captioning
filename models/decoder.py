# decoder.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, encoder_dim=768, num_layers=2):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Project ViT encoder output to decoder hidden_dim if needed
        self.encoder_projection = nn.Linear(encoder_dim, hidden_dim)

    def forward(self, input_ids, encoder_outputs, tgt_attention_mask=None):
        embedded = self.embedding(input_ids).permute(1, 0, 2)
        embedded = self.positional_encoding(embedded)

        memory = self.encoder_projection(encoder_outputs).unsqueeze(0)

        tgt_mask = generate_square_subsequent_mask(embedded.size(0)).to(embedded.device)

        if tgt_attention_mask is not None:
            tgt_key_padding_mask = ~tgt_attention_mask.bool()
        else:
            tgt_key_padding_mask = None

        output = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output).permute(1, 0, 2)
        return output

    def generate(self, encoder_outputs, tokenizer, max_length=32, sos_token_id=101, eos_token_id=102):
        generated_ids = torch.tensor([[sos_token_id]]).to(encoder_outputs.device)  # [1, 1]

        encoder_outputs_proj = self.encoder_projection(encoder_outputs)  # [batch_size, hidden_dim]
        memory = encoder_outputs_proj.unsqueeze(0)  # [1, batch_size, hidden_dim]

        for _ in range(max_length):
            tgt = self.embedding(generated_ids).permute(1, 0, 2)  # [cur_len, batch_size, hidden_dim]
            tgt = self.positional_encoding(tgt)

            tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(tgt.device).float()

            output = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )

            logits = self.fc_out(output[-1, :, :])  # [batch_size, vocab_size]

            # ✅ Greedy maximum likelihood: pick the most probable token
            next_token = logits.argmax(-1).unsqueeze(0)  # [1, batch_size]

            generated_ids = torch.cat((generated_ids, next_token.T), dim=1)

            if next_token.item() == eos_token_id:
                break

        caption = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return caption