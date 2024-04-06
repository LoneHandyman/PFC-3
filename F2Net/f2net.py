from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from tf_utils import PositionalEncoding

class F2NetHead(nn.Module):
    def __init__(self, emb_dim: int, d_model: int) -> None:
        super(F2NetHead, self).__init__()

        self.Wp = nn.Linear(emb_dim, d_model)
        self.Wa = nn.Linear(d_model, d_model)
        self.Wb = nn.Linear(d_model, d_model)

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
    def __init__(self, heads: int, emb_dim: int) -> None:
        super(F2NetMHM, self).__init__()

        d_model = emb_dim // heads
        self.heads = nn.ModuleList(
            [F2NetHead(emb_dim, d_model) for _ in range(heads)])
        self.Wmhm = nn.Linear(emb_dim, emb_dim)
        self.l_norm = nn.LayerNorm(emb_dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor):        
        x_ = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.l_norm(self.Wmhm(x_) + x)

        return out
    
class F2NetFFN(nn.Module):
    def __init__(self, emb_dim: int, hidden: int) -> None:
        super(F2NetFFN, self).__init__()

        self.W1 = nn.Linear(emb_dim, hidden)
        self.W2 = nn.Linear(hidden, emb_dim)
        self.l_norm = nn.LayerNorm(emb_dim, eps=1e-5, elementwise_affine=False)
        
    def forward(self, x: torch.Tensor):
        out1 = F.gelu(self.W1(x))
        out2 = F.gelu(self.W2(out1))
        out = self.l_norm(out2 + x)

        return out
    
class F2NetBlock(nn.Module):
    def __init__(self, heads: int, emb_dim: int, hidden: int) -> None:
        super(F2NetBlock, self).__init__()

        self.mixer = F2NetMHM(heads, emb_dim)
        self.ffn = F2NetFFN(emb_dim, hidden)

    def forward(self, x: torch.Tensor):
        x = self.mixer(x)

        out = self.ffn(x)

        return out
    
class F2NetModel(nn.Module):
    def __init__(self, n_blocks: int, heads: int, vocab_len: int, 
                 emb_dim: int, hidden: int, freeze_emb: bool = True) -> None:
        super(F2NetModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_dim)

        if freeze_emb:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.pos_encoding = PositionalEncoding(emb_dim)
        self.blocks = nn.ModuleList([
            F2NetBlock(heads, emb_dim, hidden) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.pos_encoding(self.embedding(x))

        if y is not None:
            y = self.embedding(y)

        for block in self.blocks:
            x = block(x)

        return x, y