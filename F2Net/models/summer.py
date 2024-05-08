import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.complex_scan import complex_scan

from einops import rearrange, repeat

class GlobalConv(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.d_model = kwargs['d_model']
        self.d_conv = kwargs['d_conv']
        self.scale = 1 / (self.d_model ** 0.5)

        self.from_w = nn.Linear(self.d_model, self.d_model, bias=False)
        self.conv = nn.Conv1d(self.d_model, self.d_model, self.d_conv, 
                              padding=self.d_conv // 2)
        
    def forward(self, x_to: torch.Tensor, x_from: torch.Tensor) -> torch.Tensor:
        XF = rearrange(x_from, "b n d -> b d n")
        cxf = F.silu(self.conv(XF))
        cxf = rearrange(cxf, "b d n -> b n d")
        
        s = x_to * F.softmax(self.from_w(cxf) * self.scale, dim=-2)
        glob_s = torch.sum(s, dim=-2)

        glob_s = repeat(glob_s, "b d -> b n d", n=x_to.size(-2))

        return glob_s

class Summer(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.d_model = kwargs['d_model']

        self.summ = GlobalConv(**kwargs)

        self.in_proj = nn.Linear(self.d_model, self.d_model*4)
        self.mid_proj = nn.Linear(self.d_model, self.d_model*2)
        self.out_proj = nn.Linear(2*self.d_model, self.d_model)

        nu_log, theta_log, gamma_log = self.initializer()
        self.nu_log = nn.Parameter(nu_log, requires_grad=True)
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)

        self.dropout = nn.Dropout(p=0.2)

    def initializer(self):
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.d_model)
        u2 = np.random.random(self.d_model)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(x)
        qo, g  = u.chunk(2,dim=-1)
        q, o = qo.chunk(2,dim=-1)

        nu = torch.exp(-torch.exp(self.nu_log))
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(self.gamma_log)

        f_real = nu * torch.cos(theta)
        f_imag = nu * torch.sin(theta)

        v = self.mid_proj(o * self.summ(q, q))
        
        input_real, input_imag = v.chunk(2, dim=-1)
        input_real = gamma[None, None, :] * input_real
        input_imag = gamma[None, None, :] * input_imag        
        
        f_real = f_real[None, None, :].expand_as(input_real)
        f_imag = f_imag[None, None, :].expand_as(input_real)
    
        output_real, output_imag = complex_scan(
            input_real.contiguous(), input_imag.contiguous(),
            f_real.contiguous(), f_imag.contiguous()
        )

        return self.out_proj(
            self.dropout(torch.cat([output_real, output_imag], dim=-1) * F.silu(g))
        )
    
    