import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if 'd' not in kwargs:
            kwargs['d'] = kwargs['d_model']

        self.Wq = nn.Linear(kwargs['d_model'], kwargs['d'])
        self.Wk = nn.Linear(kwargs['d_model'], kwargs['d'])
        self.Wv = nn.Linear(kwargs['d_model'], kwargs['d'])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Kt = rearrange(K, "b n d -> b d n")

        attn = torch.einsum("b m d, b d n -> b m n", Q, Kt)

        sf = 1/(Kt.size(-2)**0.5)

        scores = attn * sf

        if mask is not None:
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask==0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1)

        context = torch.einsum("b m n, b n d -> b m d", attn, V)

        return context