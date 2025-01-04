import torch
from torchvision import transforms

# from opt import MEAN, STD
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 512
size = (IMAGE_SIZE, IMAGE_SIZE)

img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=MEAN, std=STD)])

mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x

def reverse_transform(tensor, mean=MEAN, std=STD):
    # Undo the normalization by multiplying by std and adding the mean
    tensor = tensor * torch.tensor(std).view(-1, 1, 1)  # Apply std for each channel
    tensor = tensor + torch.tensor(mean).view(-1, 1, 1)  # Apply mean for each channel
    return tensor

def reverse_transform_list(tensor, mean=MEAN, std=STD):
    resutl = []
    for ts in tensor:
        # Undo the normalization by multiplying by std and adding the mean
        ts = ts * torch.tensor(std).view(-1, 1, 1)  # Apply std for each channel
        ts = ts + torch.tensor(mean).view(-1, 1, 1)  # Apply mean for each channel

        resutl.append(ts)

    return resutl