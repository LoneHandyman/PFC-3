import torch
import torch.nn as nn

class FNetTokenMixer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(FNetTokenMixer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.fft.fft(x, dim=-1)
        mix = torch.fft.fft(hidden, dim=-2)

        return torch.real(mix)
    
class HartleyTokenMixer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(HartleyTokenMixer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.fft.fft(x, dim=-1)
        mix = torch.fft.fft(hidden, dim=-2)

        return torch.real(mix) - torch.imag(mix)
    
def hartley(x: torch.Tensor, dim: int):
    ft = torch.fft.fft(x, dim=dim)
    return torch.real(ft) - torch.imag(ft)