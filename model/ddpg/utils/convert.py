import torch
from torch.autograd import Variable


def to_numpy(var):
    return var.cpu().data.numpy()

def to_tensor(ndarray, dtype=torch.float32, device='cpu'):
    return torch.tensor(ndarray).to(device).type(dtype)