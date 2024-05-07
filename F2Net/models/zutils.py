import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 512):
        super(PositionalEncoding, self).__init__()    

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, d_model)    
        k = torch.arange(0, max_length).unsqueeze(1)  

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(k * div_term)    
        pe[:, 1::2] = torch.cos(k * div_term)  

        pe = pe.unsqueeze(0)          
        self.register_buffer("pe", pe)                        

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
    
        return self.dropout(x)

from models.transformer import TransformerModel, TransformerBlock, MultiHeadModule, SingleHeadModule
from models.attention import SelfAttnHead
from models.fnet import FNetTokenMixer
from models.summer import Summer

class ModelForNextTokenPrediction(nn.Module):
    def __init__(self, encoder: nn.Module, **kwargs) -> None:
        super(ModelForNextTokenPrediction, self).__init__()

        self.model = encoder
        self.fc = nn.Linear(kwargs['d_model'], kwargs['vocab_len'], bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        logits = self.fc(x)
        
        return logits

def build_predictor(**kwargs) -> nn.Module:
    assert('feature-extractors' in kwargs)
    model = TransformerModel(**kwargs)

    for extractor in kwargs['feature-extractors']:
        id, n_blocks = extractor.split(':')

        for _ in range(int(n_blocks)):
            if id == 'attn':
                model.add_block(TransformerBlock(MultiHeadModule(SelfAttnHead, **kwargs), 
                                                 **kwargs))
            elif id == 'fnet':
                model.add_block(TransformerBlock(SingleHeadModule(FNetTokenMixer, **kwargs), 
                                                 **kwargs))
            elif id == 'summer':
                model.add_block(TransformerBlock(SingleHeadModule(Summer, **kwargs), 
                                          **kwargs))
    
    return ModelForNextTokenPrediction(model, **kwargs)