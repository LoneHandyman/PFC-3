import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset

class TextDatasetLoader:
    def __init__(self, dataset_name, dataset_config, tokenizer_name, min_freq=3, batch_size=128):
        self.dataset = load_dataset(dataset_name, dataset_config)
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.tokenized_dataset = self._tokenize_dataset()
        self.vocab = self._build_vocab(min_freq)
        self.batch_size = batch_size
        self.train_data = self._get_data('train')
        self.validation_data = self._get_data('validation')
        self.test_data = self._get_data('test')

    def _tokenize_data(self, sequence):
        tokens = self.tokenizer(sequence['text'])
        return {'tokens': tokens}

    def _tokenize_dataset(self):
        return self.dataset.map(self._tokenize_data, remove_columns=['text'])

    def _build_vocab(self, min_freq):
        tokens_iterator = (sequence['tokens'] for sequence in self.tokenized_dataset['train'])
        vocab = build_vocab_from_iterator(tokens_iterator, min_freq=min_freq)
        vocab.insert_token('<unk>', 0)
        vocab.insert_token('<eos>', 1)
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def _get_data(self, split):
        data = []
        for sequence in self.tokenized_dataset[split]:
            if sequence['tokens']:
                tokens = sequence['tokens'] + ['<eos>']
                tokens = [self.vocab[token] for token in tokens]
                data.extend(tokens)
        data = torch.LongTensor(data)
        num_batches = data.shape[0] // self.batch_size
        data = data[:num_batches * self.batch_size]
        data = data.view(self.batch_size, num_batches)
        return data

    def vocab_length(self):
        return len(self.vocab)

    def int2token(self, x: torch.LongTensor):
        itos = self.vocab.get_itos()
        tokens = []
        for t in x:
            seq = [itos[i] for i in t]
            tokens.append(seq)

        return tokens

def get_batch(data, seq_len, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target
