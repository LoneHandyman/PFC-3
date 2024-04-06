from f2net import F2NetModel
from data_builder import TextDatasetLoader, get_batch
import torch

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

print(train_data[0], train_data[0].size())

src, target = get_batch(train_data, 32, 0)

print(src.size(), target.size())
print(src, '\n', target)

model_conf = {'n_blocks':4,
              'heads':8, 
              'vocab_len': wiki2.vocab_length(), 
              'emb_dim': 128, 
              'hidden':256}

print(model_conf)

model = F2NetModel(**model_conf).to(device)

y, _ = model(src.to(device))

print(y.size())
print(y)