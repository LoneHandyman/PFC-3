import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Callable

from models.zutils import PositionalEncoding

class SingleHeadModule(nn.Module):
    def __init__(self, operationClass: Type[nn.Module], **kwargs) -> None:
        super(SingleHeadModule, self).__init__()

        self.op = operationClass(**kwargs)
        self.l_norm = nn.LayerNorm(kwargs['d_model'], eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x_ = self.op(x)

        return self.l_norm(x_ + x)

class MultiHeadModule(nn.Module):
    def __init__(self, operationClass: Type[nn.Module], **kwargs) -> None:
        super(MultiHeadModule, self).__init__()

        kwargs['d'] = kwargs['d_model'] // kwargs['heads']

        self.heads = nn.ModuleList(
            [operationClass(**kwargs) for _ in range(kwargs['heads'])])
        self.Wmhm = nn.Linear(kwargs['d_model'], kwargs['d_model'])
        self.l_norm = nn.LayerNorm(kwargs['d_model'], eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x_ = torch.cat([head(x) for head in self.heads], dim=-1)

        return self.l_norm(self.Wmhm(x_) + x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: int=0.1) -> None:
        super(FeedForward, self).__init__()

        self.W1 = nn.Linear(d_model, hidden)
        self.drop = nn.Dropout(dropout)
        self.W2 = nn.Linear(hidden, d_model)
        self.l_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor):
        out = self.W2(F.gelu(self.W1(x)))

        return self.l_norm(self.drop(out) + x)

class TransformerBlock(nn.Module):
    def __init__(self, sequence_mixer: Type[nn.Module], **kwargs) -> None:
        super(TransformerBlock, self).__init__()
        self.body = nn.ModuleList([sequence_mixer,
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

