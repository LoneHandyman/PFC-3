import torch
import torch.nn as nn
import torch.optim as optim

from models.f2net import F2NetModel
from models.f2net_lmc import F2NetForLanguageModeling
from data_builder import TextDatasetLoader
from model_trainer import Trainer

import sys

def tokenClassifier_Call(model: nn.Module, penalty: nn.CrossEntropyLoss, 
                        src: torch.LongTensor, target: torch.LongTensor,
                        device: torch.device):
    batch_size, seq_len = src.shape[0], src.shape[1]
    logits = model(src)

    prediction = logits.reshape(batch_size * seq_len, -1)   
    target = target.reshape(-1)
    loss = penalty(prediction, target)

    return loss

def noGradEmb_Call(model: F2NetModel, penalty: nn.CosineEmbeddingLoss, 
                      src: torch.LongTensor, target: torch.LongTensor,
                      device: torch.device):
    y_pred, y_true = model(src, target)
    batch_size, seq_len = y_pred.size(0), y_pred.size(1)

    y_pred = y_pred.reshape(batch_size * seq_len, -1).to(device)
    y_true = y_true.reshape(batch_size * seq_len, -1).to(device)

    loss = penalty(y_pred, y_true, torch.ones(batch_size * seq_len).to(device))

    return loss


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
        min_freq=3,
        batch_size=128
    )

    train_data = wiki2.train_data
    validation_data = wiki2.validation_data
    test_data = wiki2.test_data

    op = int(sys.argv[1])

    model_conf = {'n_blocks':2,
                  'heads':8, 
                  'vocab_len': wiki2.vocab_length(), 
                  'emb_dim': 128, 
                  'hidden':128}

    if op == 0:
        model = F2NetModel(**model_conf).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-4)
        penalty = nn.CosineEmbeddingLoss()

        trainer = Trainer('f2net-ne-(lm)-v1', model, optimizer, penalty)

        trainer.saveSettings('weights', 16)

        print('loss:', trainer.train(noGradEmb_Call, train_data, 128, 0.25, device))

    elif op == 1:
        model = F2NetForLanguageModeling(**model_conf).to(device)

        optimizer = optim.Adam(model.parameters(), 1e-4)
        penalty = nn.CrossEntropyLoss()

        trainer = Trainer('f2net-tk-(lm)-v1', model, optimizer, penalty)

        trainer.saveSettings('weights', 16)

        print('loss:', trainer.train(tokenClassifier_Call, train_data, 128, 0.25, device))