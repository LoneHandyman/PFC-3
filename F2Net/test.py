from f2net import F2NetModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = F2NetModel(4, 8, 1000, 32, 16).to(device)

x = torch.randint(0, 999, (64, 16), dtype=torch.long).to(device)

y = model(x)

print(y)