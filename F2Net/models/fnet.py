import torch
import torch.nn as nn

from models.f2net import F2NetModel

class FNetTokenMixer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.fft.fft(x, dim=-1)
        mix = torch.fft.fft(hidden, dim=-2)

        return torch.real(mix)


class FNetModel(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(FNetModel, self).__init__()

        self.model = F2NetModel(freeze_emb=False, **kwargs)

        for block in self.model.blocks:
            block.body[0] = FNetTokenMixer()

        self.fc = nn.Linear(kwargs['d_model'], kwargs['vocab_len'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.model(x)
        logits = self.fc(x)
        
        return logits