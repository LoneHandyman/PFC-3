import torch
import torch.nn as nn
import torch.optim as optim

from f2net import F2NetModel
from data_builder import TextDatasetLoader
from model_trainer import Trainer

def callable_on_train(model: nn.Module, penalty: nn.Module, 
                      src: torch.LongTensor, target: torch.LongTensor,
                      device: torch.device):
    y_pred, y_true = model(src, target)
    batch_size, seq_len = y_pred.size(0), y_pred.size(1)

    y_pred = y_pred.reshape(batch_size * seq_len, -1).to(device)
    y_true = y_true.reshape(batch_size * seq_len, -1).to(device)

    loss = penalty(y_pred, y_true, torch.ones(batch_size * seq_len).to(device))

    return loss


if __name__ == '__main__':
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wiki2 = TextDatasetLoader(
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1',
        tokenizer_name='basic_english',
        min_freq=3,
        batch_size=64
    )

    train_data = wiki2.train_data
    validation_data = wiki2.validation_data
    test_data = wiki2.test_data

    model_conf = {'n_blocks':4,
                  'heads':8, 
                  'vocab_len': wiki2.vocab_length(), 
                  'emb_dim': 128, 
                  'hidden':256}

    model = F2NetModel(**model_conf).to(device)

    optimizer = optim.Adam(model.parameters(), 1e-4)
    penalty = nn.CosineEmbeddingLoss()

    trainer = Trainer(model, optimizer, penalty)

    trainer.saveSettings('weights', 16)

    print('loss:', trainer.train(callable_on_train, train_data, 128, 0.25, device))
