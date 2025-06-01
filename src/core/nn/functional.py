import torch
from ..tensor import Tensor

def relu(x):
    return Tensor(torch.maximum(x.data, torch.tensor(0.)))

def sigmoid(x):
    return Tensor(1 / (1 + torch.exp(-x.data)))

def tanh(x):
    return Tensor(torch.tanh(x.data))