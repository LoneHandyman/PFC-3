import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

from data_builder import TextDatasetLoader
from model_trainer import Trainer, model_factory

from model_metrics import *

import sys
import os

def tokenClassifier_Call(model: nn.Module, penalty: nn.CrossEntropyLoss, 
                        src: torch.LongTensor, target: torch.LongTensor,
                        device: torch.device):
    batch_size, seq_len = src.shape[0], src.shape[1]
    logits = model(src)

    prediction = logits.reshape(batch_size * seq_len, -1)   
    target_r = target.reshape(-1)
    loss = penalty(prediction, target_r)

    return loss, logits, target

def noGradEmb_Call(model: nn.Module, penalty, src: torch.LongTensor, 
                   target: torch.LongTensor, device: torch.device):
    y_pred, y_true = model(src, target)
    batch_size, seq_len = y_pred.size(0), y_pred.size(1)

    y_pred = y_pred.reshape(batch_size * seq_len, -1).to(device)
    y_true = y_true.reshape(batch_size * seq_len, -1).to(device)

    if isinstance(penalty, nn.CosineEmbeddingLoss):
        loss = penalty(y_pred, y_true, torch.ones(batch_size * seq_len).to(device))
    elif isinstance(penalty, nn.KLDivLoss):
        loss = penalty(y_pred, y_true)
    return loss, y_pred, y_true

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Zero arguments provided.')
        exit(0)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wiki2 = TextDatasetLoader(
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1',
        tokenizer_name='basic_english',
        min_freq=12,
        batch_size=64
    )

    train_data = wiki2.train_data
    validation_data = wiki2.validation_data

    op = int(sys.argv[1])

    name, model, lr = model_factory('models/config.json', op, wiki2.vocab_length())

    wdir = 'weights'

    if not os.path.exists(wdir):
        os.makedirs(wdir)

    if len(sys.argv) == 3 and sys.argv[2] == '-bkp':
        model.load_state_dict(torch.load(wdir + '/' + name + '.pt'))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                                                            patience=0)
    trainer = Trainer(name, model, optimizer, nn.CrossEntropyLoss(), lr_scheduler)

    trainer.saveSettings(wdir)
    trainer.setDataLoaders(train_data, validation_data)
    trainer.setFeedForwardProcedure(tokenClassifier_Call)

    trainer.setMetrics([mPerplexity, mAccuracyF1])

    try:
        trainer.train(seq_len=128, n_epochs=100, clip=0.25, device=device)
    except KeyboardInterrupt:
        print("\n[STATUS]: Keyboard interruption.")
        sys.exit(0)