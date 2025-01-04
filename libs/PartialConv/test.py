import torch
from PIL import Image
from torchvision.utils import save_image

from util.display import display
from model.model import PConvUNet
from util.transform import reverse_transform, unnormalize, img_tf, mask_tf
from util.io import load_ckpt

import gc
torch.cuda.empty_cache()
gc.collect()

image_size = 512
data_root = '../../../dataset/img_align_celeba_modify'
masks_path = 'masks'
snapshot = '../../../pretrain_checkpoint/checkpoint-Partial-Conv-10000.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_example(image_path, mask_path, img_transform, mask_transform):

    # get model
    model = PConvUNet().to(device)
    load_ckpt(snapshot, [('model', model)])

    # load image
    gt_img = Image.open(image_path)
    mask_img = Image.open(mask_path)
    gt = img_transform(gt_img.convert('RGB'))
    mask = mask_transform(mask_img.convert('RGB'))

    # create image with mask - save to .data
    masked_img = gt * mask
    masked_save = reverse_transform(masked_img)
    save_image(masked_save, '../../data/masked_img.png')

    # gt_img = torch.tensor(gt).unsqueeze(0)  # Add batch dimension (assuming image is in HWC format)
    mask = torch.tensor(mask).unsqueeze(0)
    masked_img = torch.tensor(masked_img).unsqueeze(0)

    # generate output
    with torch.no_grad():
        output, _ = model(masked_img.to(device), mask.to(device))

    # convert output to image and save
    output = output.to(torch.device('cpu'))
    output = unnormalize(output)
    save_image(output, '../../data/output.png')

    # matplotlib display
    masked_save = masked_save.permute(1, 2, 0).detach().cpu().numpy()
    output = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    display(gt_img, mask_img, masked_save, output)


if __name__ == "__main__":
    image_path = '../../data/messi.jpg'
    masks_path = '../../masks/mask_1/000001.jpg'

    test_example(image_path, masks_path, img_tf, mask_tf)