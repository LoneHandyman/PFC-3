import torch
import torch.nn as nn
import math
from tqdm import tqdm

from typing import Callable, Tuple
from data_builder import get_batch

class Trainer:
    def __init__(self, name, model, optimizer, penalty, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.penalty = penalty
        self.lr_scheduler = lr_scheduler

        self.train_data = None
        self.eval_data = None

        self.fefo_steps = None
        self.metric_steps = []
        self.metric_results = {}

        self.name = name
        self.epochs2save = 0
        self.path = ''

    def setDataLoaders(self, traind: torch.LongTensor, evald: torch.LongTensor):
        self.train_data = traind
        self.eval_data = evald

    def setFeedForwardProcedure(self, fefo_steps: Callable[[nn.Module, nn.Module, 
                                                            torch.LongTensor, 
                                                            torch.LongTensor, torch.device], 
                                                            Tuple[torch.Tensor, torch.Tensor, 
                                                                  torch.Tensor]]):
        self.fefo_steps = fefo_steps

    def setMetrics(self, metric_list):
        self.metric_steps = metric_list

    def saveSettings(self, path: str):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'

    def _check_params(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'The model [{self.name}] has {num_params:,} trainable parameters')

    def evaluate(self, seq_len: int, device: torch.device):
        assert(self.eval_data is not None)
        assert(self.fefo_steps is not None)

        epoch_loss = 0
        self.model.eval()

        num_batches = self.eval_data.shape[-1]
        data = self.eval_data[:, :num_batches - (num_batches -1) % seq_len]
        num_batches = data.shape[-1]

        with torch.no_grad():
            for idx in range(0, num_batches - 1, seq_len):
                src, target = get_batch(data, seq_len, idx)
                src, target = src.to(device), target.to(device)

                loss, pred, target = self.fefo_steps(self.model, self.penalty, src, target, device)

                epoch_loss += loss.item() * seq_len

                for metric in self.metric_steps:
                    metric(pred, target, device, self.metric_results)

        for key, _ in self.metric_results.items():
            self.metric_results[key] /= num_batches

        return epoch_loss / num_batches

    def fit(self, seq_len: int, clip: float, device: torch.device, 
            epoch_step: Tuple[int, int]):
        assert(self.train_data is not None)
        assert(self.fefo_steps is not None)

        epoch_loss = 0
        self.model.train()

        num_batches = self.train_data.shape[-1]
        data = self.train_data[:, :num_batches - (num_batches -1) % seq_len]
        num_batches = data.shape[-1]
        
        train_bar = tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False)

        train_bar.set_description_str(f"Epoch[{epoch_step[0]}/{epoch_step[1]}]")

        for idx in train_bar:
            self.optimizer.zero_grad()

            src, target = get_batch(data, seq_len, idx)
            src, target = src.to(device), target.to(device)

            loss, pred, target = self.fefo_steps(self.model, self.penalty, src, target, device)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            epoch_loss += loss.item() * seq_len

            train_bar.set_postfix_str(f"Loss: {loss.item():.4f}")
            
        return epoch_loss / num_batches
    
    def train(self, seq_len: int, n_epochs: int, clip: float, device: torch.device):
        self._check_params()

        best_eval_loss = float('inf')

        self.model.to(device)

        for epoch in range(n_epochs):
            train_loss = self.fit(seq_len, clip, device, (epoch+1, n_epochs))
            eval_loss = self.evaluate(seq_len, device)
            
            self.lr_scheduler.step(eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(self.model.state_dict(), self.path + self.name + '.pt')

            print(f'Completed epoch #{epoch+1}:')
            for key, result in self.metric_results.items():
                print(f'\tMetric[{key}] = {result:.4f}')
            #print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
            #print(f'\tValid Perplexity: {math.exp(eval_loss):.3f}')