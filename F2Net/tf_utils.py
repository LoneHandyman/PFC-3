import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, emb_dim: int, dropout: float = 0.1, max_length: int = 5000):
    super(PositionalEncoding, self).__init__()    

    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_length, emb_dim)    
    k = torch.arange(0, max_length).unsqueeze(1)  

    div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))

    pe[:, 0::2] = torch.sin(k * div_term)    
    pe[:, 1::2] = torch.cos(k * div_term)  

    pe = pe.unsqueeze(0)          
    self.register_buffer("pe", pe)                        

  def forward(self, x: torch.Tensor):
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
    
    return self.dropout(x)