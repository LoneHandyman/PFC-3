from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from tf_utils import PositionalEncoding

class F2NetHead(nn.Module):
    def __init__(self, d_model: int, d: int) -> None:
        super(F2NetHead, self).__init__()

        self.Wp = nn.Linear(d_model, d)
        self.Wa = nn.Linear(d, d)
        self.Wb = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        #x : [B, S, E]
        p = self.Wp(x)

        features = torch.fft.fft(p, dim=-1)

        r = torch.real(features)
        i = torch.imag(features)

        rr = r + self.Wa(r)
        ii = i + self.Wb(i)
        fixed = torch.complex(rr, ii)

        seq = torch.fft.ifft(fixed, dim=-2)

        return torch.real(seq)

class F2NetMHM(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super(F2NetMHM, self).__init__()

        d = d_model // heads
        self.heads = nn.ModuleList(
            [F2NetHead(d_model, d) for _ in range(heads)])
        self.Wmhm = nn.Linear(d_model, d_model)
        self.l_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor):        
        x_ = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.l_norm(self.Wmhm(x_) + x)

        return out
    
class F2NetFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super(F2NetFFN, self).__init__()

        self.W1 = nn.Linear(d_model, hidden)
        self.W2 = nn.Linear(hidden, d_model)
        self.l_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor):
        out1 = F.gelu(self.W1(x))
        out2 = F.gelu(self.W2(out1))
        out = self.l_norm(out2 + x)

        return out
    
class F2NetBlock(nn.Module):
    def __init__(self, heads: int, d_model: int, hidden: int) -> None:
        super(F2NetBlock, self).__init__()

        self.body = nn.ModuleList([F2NetMHM(heads, d_model),
                                   F2NetFFN(d_model, hidden)
        ])

    def forward(self, x: torch.Tensor):
        for module in self.body:
            x = module(x)

        return x
    
class F2NetModel(nn.Module):
    def __init__(self, n_blocks: int, heads: int, vocab_len: int, 
                 d_model: int, hidden: int, freeze_emb: bool = True) -> None:
        super(F2NetModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, d_model)

        if freeze_emb:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            F2NetBlock(heads, d_model, hidden) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.pos_encoding(self.embedding(x))

        if y is not None:
            y = self.embedding(y)

        for block in self.blocks:
            x = block(x)

        return x, y
    
class F2Net2Vocab(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(F2Net2Vocab, self).__init__()

        self.model = F2NetModel(freeze_emb=False, **kwargs)
        self.fc = nn.Linear(kwargs['d_model'], kwargs['vocab_len'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.model(x)
        logits = self.fc(x)
        
        return logits