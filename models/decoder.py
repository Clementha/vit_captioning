# decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.vocab_size = vocab_size
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

    def generate(
        self,
        encoder_outputs,
        start_token_id=101,  # [CLS] token for BERT
        eos_token_id=102,
        max_length=50,
        mode="greedy",      # "greedy", "beam", "topk", "topp"
        num_beams=3,
        top_k=50,
        top_p=0.95,
        length_penalty=1.0
    ):
        
        device = encoder_outputs.device

        """
        Generate caption using specified decoding mode.
        """
        batch_size = encoder_outputs.size(0)
        input_ids = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=device
        )

        if mode == "beam":
            return self._generate_beam_search(
                encoder_outputs,
                input_ids,
                max_length,
                eos_token_id,
                num_beams,
                length_penalty
            )

        # Greedy or sampling
        generated = input_ids

        for _ in range(max_length):
            logits = self.forward(generated, encoder_outputs)   # (batch, seq_len, vocab)
            next_token_logits = logits[:, -1, :]                # (batch, vocab)

            if mode == "greedy":
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            elif mode == "topk":
                probs = F.softmax(next_token_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, top_k)
                next_token = topk_indices[
                    torch.arange(probs.size(0)),
                    torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
                ].unsqueeze(-1)

            elif mode == "topp":
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probs above threshold
                sorted_mask = cumulative_probs <= top_p
                sorted_mask[..., 0] = 1  # Always keep at least 1 token

                filtered_probs = sorted_probs * sorted_mask
                filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

                next_token = sorted_indices[
                    torch.arange(probs.size(0)),
                    torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
                ].unsqueeze(-1)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated[:, 1:]  # Remove BOS if needed

    def _generate_beam_search(
        self,
        encoder_outputs,
        input_ids,
        max_length=50,
        eos_token_id=102,
        num_beams=3,
        length_penalty=1.0
    ):
        """
        Custom beam search decoder for batch_size = 1.
        """
        device = encoder_outputs.device
        batch_size = encoder_outputs.size(0)
        vocab_size = self.vocab_size

        # Assume batch_size = 1 for simplicity
        assert batch_size == 1, "Basic beam search only supports batch size 1 here."

        # Initialize beams
        beam_sequences = [input_ids] * num_beams
        beam_scores = torch.zeros(num_beams, device=device)

        finished_sequences = []
        finished_scores = []

        for step in range(max_length):
            all_candidates = []

            for beam_idx in range(num_beams):
                seq = beam_sequences[beam_idx]
                score = beam_scores[beam_idx]

                logits = self.forward(seq, encoder_outputs)  # (1, seq_len, vocab)
                next_token_logits = logits[:, -1, :]         # (1, vocab)
                log_probs = F.log_softmax(next_token_logits, dim=-1).squeeze(0)  # (vocab,)

                for token_id in range(vocab_size):
                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    new_score = score + log_probs[token_id]
                    all_candidates.append((new_seq, new_score))

            # Get top beams
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam_sequences = []
            beam_scores = []

            for seq, score in all_candidates[:num_beams]:
                if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                    finished_sequences.append(seq)
                    finished_scores.append(score)
                else:
                    beam_sequences.append(seq)
                    beam_scores.append(score)

            beam_scores = torch.stack(beam_scores) if beam_scores else torch.tensor([], device=device)

            # Early stopping if all beams ended
            if len(beam_sequences) == 0:
                break

        # Add unfinished beams to finished
        if not finished_sequences:
            finished_sequences = beam_sequences
            finished_scores = beam_scores

        # Length penalty
        finished_scores = [s / (len(seq[0]) ** length_penalty) for seq, s in zip(finished_sequences, finished_scores)]

        # Pick best
        best_idx = torch.tensor(finished_scores).argmax().item()
        best_seq = finished_sequences[best_idx]

        return best_seq[:, 1:]  # remove BOS if needed
