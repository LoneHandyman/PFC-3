import torch
import torch.nn as nn
from tqdm import tqdm

from typing import Callable
from data_builder import get_batch

class Trainer:
    def __init__(self, model, optimizer, penalty):
        self.model = model
        self.optimizer = optimizer
        self.penalty = penalty
        self.epochs2save = 0
        self.path = ''

    def loadFromFile(self, path):
        pass

    def saveSettings(self, path, epochs2save):
        self.path = path
        self.epochs2save = epochs2save

    def _check_params(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'The model has {num_params:,} trainable parameters')

    def train(self, fefo_steps: Callable[[nn.Module, nn.Module, torch.LongTensor, 
                                          torch.LongTensor, torch.device], torch.Tensor],
                                          data: torch.LongTensor, seq_len: int, 
                                          clip: float, device: torch.device):
        self._check_params()

        epoch_loss = 0

        self.model.train()

        num_batches = data.shape[-1]
        data = data[:, :num_batches - (num_batches -1) % seq_len]
        num_batches = data.shape[-1]
        
        for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):
            self.optimizer.zero_grad()

            src, target = get_batch(data, seq_len, idx)
            src, target = src.to(device), target.to(device)

            loss = fefo_steps(self.model, self.penalty, src, target, device)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            epoch_loss += loss.item() * seq_len
            
        return epoch_loss / num_batches