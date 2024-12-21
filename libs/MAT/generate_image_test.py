
"""Generate images using pretrained network pickle."""
import os
import random

import PIL.Image
import cv2
import numpy as np
import pyspng
import torch

import dnnlib
import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def generate_image(
        network_pkl: str,
        resolution: str,
        truncation_psi: str,
        noise_mode: str,
        mask_threshold: str,
        ipath: str,
        mpath: str):

    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
    net_res = 512 if resolution > 512 else resolution
    model = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(
        device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, model, require_all=True)

    label = torch.zeros([1, model.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        image = image[:3]
        return image

    def read_mask(mask_path):
        return cv2.imread(mpath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / mask_threshold

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    # img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if resolution != 512:
        noise_mode = 'random'

    with torch.no_grad():

        iname = os.path.basename(ipath).replace('.jpg', '.png')
        print(f'Prcessing: {iname}')

        # image = cv2.imread(ipath)
        # image = cv2.resize(image, (512, 512))
        # cv2.imwrite('./testImage/mask_image.png', image)
        #
        # mask = cv2.imread(mpath)
        # mask = cv2.resize(mask, (512, 512))
        # cv2.imwrite('./testImage/mask.png', mask)

        image = read_image(ipath)
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

        if mpath is not None:
            mask = read_mask(mpath)
            mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
        else:
            mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
            mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)
            cv2.imwrite('../../data/mask_test.png', mask)

        # z = torch.from_numpy(np.random.randn(1, model.z_dim)).to(device)
        # output = model(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        # output = output[0].cpu().numpy()
        # PIL.Image.fromarray(output, 'RGB').save('../../data/output.png')

if __name__ == "__main__":
    generate_image(
        network_pkl='../../../pretrain_checkpoint/CelebA-HQ_512.pkl',
        resolution = 512,
        truncation_psi = 1,
        noise_mode = "const",
        mask_threshold = 1,
        ipath = '../../data/mask_image.png',
        mpath=None
        # mpath = '../../data/mask.png',
    )

















