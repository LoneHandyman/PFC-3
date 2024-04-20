import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

from models.zutils import PositionalEncoding

class MultiHeadModule(nn.Module):
    def __init__(self, operationClass: Type[nn.Module], heads: int, d_model: int) -> None:
        super(MultiHeadModule, self).__init__()

        d = d_model // heads
        self.heads = nn.ModuleList(
            [operationClass(d_model, d) for _ in range(heads)])
        self.Wmhm = nn.Linear(d_model, d_model)
        self.l_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor):        
        x_ = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.l_norm(self.Wmhm(x_) + x)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: int=0.1) -> None:
        super(FeedForward, self).__init__()

        self.W1 = nn.Linear(d_model, hidden)
        self.drop = nn.Dropout(dropout)
        self.W2 = nn.Linear(hidden, d_model)
        self.l_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor):
        out1 = F.gelu(self.W1(x))
        out2 = self.W2(self.drop(out1))
        out = self.l_norm(out2 + x)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, feature_extractor: Type[nn.Module], **kwargs) -> None:
        super(TransformerBlock, self).__init__()

        self.body = nn.ModuleList([MultiHeadModule(feature_extractor, 
                                                   kwargs['heads'], 
                                                   kwargs['d_model']),
                                   FeedForward(kwargs['d_model'], kwargs['hidden'])
        ])

    def forward(self, x: torch.Tensor):
        for module in self.body:
            x = module(x)

        return x

class TransformerModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(kwargs['vocab_len'], kwargs['d_model'])

        self.pos_encoding = PositionalEncoding(kwargs['d_model'])
        self.blocks = nn.ModuleList()

    def add_block(self, block: TransformerBlock):
        self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.embedding(x))

        for block in self.blocks:
            x = block(x)

        return x

