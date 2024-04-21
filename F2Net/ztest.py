import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from data_builder import TextDatasetLoader
from model_tester import generate
from model_trainer import model_factory

import sys

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

    test_data = wiki2.test_data

    op = int(sys.argv[1])

    name, model, lr = model_factory('models/config.json', op, wiki2.vocab_length())

    model.load_state_dict(torch.load('weights/' + name + '.pt'))

    prompt = 'Think about me and my life. I want'
    max_seq_len = 30
    seed = 0

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, wiki2.tokenizer, 
                            wiki2.vocab, device, seed)
        print(str(temperature)+'\n'+' '.join(generation)+'\n')