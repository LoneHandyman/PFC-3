import torch
import torch.nn as nn
from torch.nn import functional as F

from models.fnet import hartley

class F2NetHead(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(F2NetHead, self).__init__()
        d_model = kwargs['d_model']
        d = d_model
        self.Wq = nn.Linear(d_model, d)
        self.Wv = nn.Linear(d_model, d)
        self.Wg = nn.Linear(d_model, d)

        self.W_a = nn.Linear(d, d, bias=False)

        self.Wout = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        V = self.Wv(hartley(x, dim=-1))
        G = self.Wg(x)

        scale_factor = 1 / (Q.size(-1) ** 0.5)

        q = Q * F.softmax(self.W_a(Q) * scale_factor, dim=-2)
        glob_q = torch.sum(q, dim=-2).unsqueeze(-2).expand(*Q.size())#nxd

        P = glob_q * V

        R = self.Wout(P * F.silu(G))

        return R