import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

from models.f2net import F2NetModel, F2Net2Vocab
from models.fnet import FNetModel
from models.transformer import TransformerModel

from data_builder import TextDatasetLoader
from model_trainer import Trainer

from model_metrics import mPerplexity, mAccuracyF1

import sys
import json

def tokenClassifier_Call(model: nn.Module, penalty: nn.CrossEntropyLoss, 
                        src: torch.LongTensor, target: torch.LongTensor,
                        device: torch.device):
    batch_size, seq_len = src.shape[0], src.shape[1]
    logits = model(src)

    prediction = logits.reshape(batch_size * seq_len, -1)   
    target_r = target.reshape(-1)
    loss = penalty(prediction, target_r)

    return loss, logits, target

def noGradEmb_Call(model: F2NetModel, penalty: nn.CosineEmbeddingLoss, 
                      src: torch.LongTensor, target: torch.LongTensor,
                      device: torch.device):
    y_pred, y_true = model(src, target)
    batch_size, seq_len = y_pred.size(0), y_pred.size(1)

    y_pred = y_pred.reshape(batch_size * seq_len, -1).to(device)
    y_true = y_true.reshape(batch_size * seq_len, -1).to(device)

    loss = penalty(y_pred, y_true, torch.ones(batch_size * seq_len).to(device))

    return loss, y_pred, y_true

def model_factory(config_path: str, op: int, vocab_len: int):
    with open(config_path, 'r') as fjsn:
        gConfig = json.load(fjsn)
    
    if op == 0:
        config = gConfig['f2net']
        return 'f2net-nge-(lm)-v1', F2NetModel(vocab_len=vocab_len, **config), nn.CosineEmbeddingLoss(), noGradEmb_Call, 1e-4
    elif op==1:
        config = gConfig['f2net']
        return 'f2net-tkc-(lm)-v1', F2Net2Vocab(vocab_len=vocab_len, **config), nn.CrossEntropyLoss(), tokenClassifier_Call, 1e-4
    elif op==2:
        config = gConfig['fnet']
        return 'fnet-tkc-(lm)', FNetModel(vocab_len=vocab_len, **config), nn.CrossEntropyLoss(), tokenClassifier_Call, 1e-4
    elif op==3:
        config = gConfig['transformer']
        return 'tfnet-tkc-(lm)', TransformerModel(vocab_len=vocab_len, **config), nn.CrossEntropyLoss(), tokenClassifier_Call, 1e-5


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Zero arguments provided.')
        exit(0)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')

    wiki2 = TextDatasetLoader(
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1',
        tokenizer_name='basic_english',
        min_freq=30,
        batch_size=128
    )

    train_data = wiki2.train_data
    validation_data = wiki2.validation_data
    test_data = wiki2.test_data

    op = int(sys.argv[1])

    name, model, penalty, fefo, lr = model_factory('models/config.json', op, wiki2.vocab_length())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                                                            patience=0)
    trainer = Trainer(name, model, optimizer, penalty, lr_scheduler)

    trainer.saveSettings('weights')
    trainer.setDataLoaders(train_data, validation_data)
    trainer.setFeedForwardProcedure(fefo)
    trainer.setMetrics([mPerplexity, mAccuracyF1])

    trainer.train(seq_len=128, n_epochs=100, clip=0.25, device=device)