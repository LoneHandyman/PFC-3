import torch
import torch.nn as nn

from models.f2net import F2NetModel

class F2NetForLanguageModeling(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(F2NetForLanguageModeling, self).__init__()

        self.model = F2NetModel(freeze_emb=False, **kwargs)
        self.fc = nn.Linear(kwargs['emb_dim'], kwargs['vocab_len'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.model(x)
        logits = self.fc(x)
        
        return logits