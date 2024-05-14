import torch
import torch.nn as nn
from torch.nn import functional as F

from models.fnet import hartley

class F2NetHead(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(F2NetHead, self).__init__()
        self.d_model = kwargs['d_model']
        self.d = self.d_model

        if 'd' in kwargs:
            self.d = kwargs['d']

        d_conv = kwargs['d_conv']

        self.Wqog = nn.Linear(self.d_model, self.d * 3)

        self.w_a = nn.Linear(self.d, self.d, bias=False)
        #self.w_b = nn.Linear(self.d, self.d, bias=False)

        self.Wout = nn.Linear(self.d, self.d)
        self.conv = nn.Conv1d(self.d, self.d, d_conv, padding=d_conv // 2)

    def summ(self, x_to: torch.Tensor, x_from: torch.Tensor, w: nn.Linear, scale_factor: int) -> torch.Tensor:
        s = x_to * F.softmax(w(x_from) * scale_factor, dim=-2)
        glob_s = torch.sum(s, dim=-2).unsqueeze(-2).expand(*x_to.size())

        return glob_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qog = self.Wqog(x)
        (Q, O, G) = qog.split(split_size=self.d, dim=-1)

        scale_factor = 1 / (Q.size(-1) ** 0.5)

        cq = F.silu(self.conv(Q.transpose(-2,-1)).transpose(-2,-1))

        P = O * self.summ(Q, cq, self.w_a, scale_factor)

        L = F.silu(G) * torch.cumsum(P, dim=-2)#self.summ(P, F.gelu(P), self.w_b, scale_factor)

        R = self.Wout(L)

        return R