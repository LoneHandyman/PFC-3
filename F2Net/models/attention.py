import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttnHead(nn.Module):
    def __init__(self, d_model: int, d: int) -> None:
        super(SelfAttnHead, self).__init__()
        self.Wq = nn.Linear(d_model, d)
        self.Wk = nn.Linear(d_model, d)
        self.Wv = nn.Linear(d_model, d)

    def forward(self, x: torch.Tensor):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        scale_factor = 1/(self.Wk.out_features**0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor

        tril = torch.tril(scores)
        attn = F.softmax(tril.masked_fill(tril==0, -1e8), dim=-1)
        context = torch.matmul(attn, V)

        return context