import torch
import time

import matplotlib.pyplot as plt
import numpy as np
import math

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.to(device)
    model.eval()

    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction = model(src)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def eval_complexity(model, input: torch.Tensor, min_seq_len: int, 
                    max_seq_len: int, step: int, device):
    model.to(device)
    model.eval()

    keys = ['tiempo[ms]', 'memoria[gb]', 'memoria(log n)[gb]']

    measures = {keys[0]:[], keys[1]:[], keys[2]:[]}

    first = True

    for seq_len in range(min_seq_len, max_seq_len, step):
        x = input.expand(input.size(-3), seq_len, input.size(-1)).to(device)

        if first:
            first = False
            model(x)

        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        model(x)

        total_time = (time.time() - start_time) * 1000
        total_time = round(total_time, 3)
        peak_memory = torch.cuda.max_memory_allocated(device)

        peak_gb = peak_memory / (1024**3)
        peak_gb = round(peak_gb, 3)
        measures[keys[0]].append(total_time)
        measures[keys[1]].append(peak_gb)
        measures[keys[2]].append(round(math.log2(peak_gb), 8))
    
    return measures

def plot_eval_data(dict_data, seq_params):
    l, r, step = seq_params
    seq_values = np.linspace(l, r, int((r - l) / step))

    for data_name in next(iter(dict_data.values())):
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name, model_data in dict_data.items():
            ax.plot(seq_values, model_data[data_name], label=model_name)
        ax.set_xlabel('Nro. tokens por secuencia')
        ax.set_ylabel(data_name)
        ax.legend()
        plt.show()

def model_set_evaluator(models, size, seq_range, device):

    evals = {}
    l, r = seq_range[0], seq_range[1]
    step = int((r - l) * 0.02)

    for id, model in models:
        x = torch.randn(size[0], 1, size[1])
        evals[id] = eval_complexity(model, x, l, r, step, device)

    print(evals)

    plot_eval_data(evals, (l, r, step))