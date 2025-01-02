import torch

from libs.PartialConv.util import opt


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x

def reverse_transform(tensor, mean, std):
    # Undo the normalization by multiplying by std and adding the mean
    tensor = tensor * torch.tensor(std).view(-1, 1, 1)  # Apply std for each channel
    tensor = tensor + torch.tensor(mean).view(-1, 1, 1)  # Apply mean for each channel
    return tensor