import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy, F1Score
from torchmetrics.text import Perplexity

def adderIn(dest, key, value):
    if key not in dest:
        dest[key] = value
    else:
        dest[key] += value

def mPerplexity(logits, target, device, dest=None):
    metric = Perplexity().to(device)

    seq_len = logits.size(-2)

    metric.update(logits.to(device), target.to(device))
    ppl = metric.compute()

    if dest is not None:
        adderIn(dest, 'ppl', ppl.item() * seq_len) 

def mAccuracyF1(logits, target, device, dest=None):
    acc_metric = Accuracy(task="multiclass", num_classes=logits.size(-1)).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=logits.size(-1)).to(device)

    probs = F.softmax(logits, dim=-1)
    pTokens = torch.argmax(probs, dim=-1)

    acc_metric.update(pTokens.to(device), target.to(device))
    f1_metric.update(pTokens.to(device), target.to(device))
    acc = acc_metric.compute()
    f1 = f1_metric.compute()

    if dest is not None:
        adderIn(dest, 'acc', acc.item())
        adderIn(dest, 'f1', f1.item())