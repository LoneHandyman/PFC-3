import torch
import torch.nn as nn

from models.zutils import PositionalEncoding

import math

class TransformerModel(nn.Module):
    def __init__(self, n_blocks: int, heads: int, vocab_len: int, 
                 d_model: int, hidden: int, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, heads, hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_blocks)
        self.encoder = nn.Embedding(vocab_len, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_len)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(-1) != src.size(-1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(-1)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

