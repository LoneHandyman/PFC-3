import torch
import torch.nn as nn
from torch.nn import functional as F

from models.fnet import hartley

class F2NetHead(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(F2NetHead, self).__init__()
        d_model = kwargs['d_model']
        d = d_model
        d_conv = 3

        self.Wq = nn.Linear(d_model, d)
        self.Wo = nn.Linear(d_model, d)
        self.Wg = nn.Linear(d_model, d)

        self.w_a = nn.Linear(d, d, bias=False)
        self.w_b = nn.Linear(d, d, bias=False)

        self.Wout = nn.Linear(d, d_model)
        self.conv = nn.Conv1d(d, d, d_conv, padding=d_conv // 2)

    def summ(self, x_to: torch.Tensor, x_from: torch.Tensor, w: nn.Linear, scale_factor: int) -> torch.Tensor:
        s = x_to * F.softmax(w(F.gelu(x_from)) * scale_factor, dim=-2)
        glob_s = torch.sum(s, dim=-2).unsqueeze(-2).expand(*x_to.size())

        return glob_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        O = self.Wo(x)
        G = self.Wg(x)

        scale_factor = 1 / (Q.size(-1) ** 0.5)

        cq = F.silu(self.conv(Q.transpose(-2,-1)).transpose(-2,-1))

        P = O * self.summ(Q, cq, self.w_a, scale_factor)#Q, G

        L = F.silu(G) * self.summ(P, P, self.w_b, scale_factor)

        R = self.Wout(L)

        return R