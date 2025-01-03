import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from PIL import Image
import cv2

from libs.PartialConv.util.transform import unnormalize, reverse_transform


def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(1000, 1008)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_fid(model, image_path, mask_path, img_transform, mask_transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



