import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from data_builder import TextDatasetLoader
from model_tester import generate
from model_trainer import Trainer, model_factory, tokenClassifier_Call
from model_metrics import mAccuracyF1, mPerplexity

import sys

def predict_next_tokens(model, prompt, max_seq_len, dataloader, seed, device):
    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for t in temperatures:
        generation = generate(prompt, max_seq_len, t, model, dataloader.tokenizer, 
                            dataloader.vocab, device, seed)
        print(str(t)+'\n'+' '.join(generation)+'\n')

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
        batch_size=16
    )

    test_data = wiki2.test_data

    op = int(sys.argv[1])

    name, model, lr = model_factory('models/config.json', op, wiki2.vocab_length())

    model.load_state_dict(torch.load('weights/' + name + '.pt'))

    model.to(device)

    evaluator = Trainer(name, model, None, nn.CrossEntropyLoss(), None)
    evaluator.setDataLoaders(None, test_data)
    evaluator.setFeedForwardProcedure(tokenClassifier_Call)
    evaluator.setMetrics([mPerplexity, mAccuracyF1])

    test_loss = evaluator.evaluate(seq_len=128, device=device)
    
    print(f'loss:{test_loss:.3f}')
    print(evaluator.metric_results)

    prompts = [
        'Think about me and',
        'Consider the possibilities, where the',
        'Reflect on love and its',
        'Imagine a world without limits',
        'Contemplate the mysteries of the'
    ]

    for p in prompts:
        predict_next_tokens(model, p, max_seq_len=46, 
                            dataloader=wiki2, seed=0, device=device)