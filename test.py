
from tools import check_gpu
import torch

print(torch.cuda.is_available())

print(check_gpu())