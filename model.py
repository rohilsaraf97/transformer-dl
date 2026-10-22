import math

import torch
import torch.nn as nn


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    Built from nn.TransformerDecoderLayer. A decoder layer has self-attention
    plus cross-attention to a "memory"; since there is no encoder here we feed
    the token embeddings themselves as memory and apply the same causal mask
    to both attentions, so no position can see future tokens.
    """

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.1, context_len=256):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tok_emb.weight  # weight tying
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (batch, seq_len) token ids -> logits (batch, seq_len, vocab)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        h = self.drop(self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        memory = h
        for layer in self.layers:
            h = layer(h, memory, tgt_mask=mask, memory_mask=mask,
                      tgt_is_causal=True, memory_is_causal=True)
        return self.head(self.norm(h))
