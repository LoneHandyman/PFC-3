import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from model_tester import model_set_evaluator

from models.attention import SelfAttention
from models.summer import Summer

if __name__ == '__main__':

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        'd_model': 256,
        'd_conv': 5
    }

    models = [('Transformer', SelfAttention(**kwargs)), 
              ('SummeRNet', Summer(**kwargs))]

    model_set_evaluator(models, (1, kwargs['d_model']), (192, 8192), device)