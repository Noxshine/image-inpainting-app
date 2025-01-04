import torch
from exceptiongroup import catch
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.display import display
from metrics.metric_utils import calculate_fid
from model.model import PConvUNet
from util.sampler import InfiniteSampler
from util.io import load_ckpt
from util.inpainting_dataset import InpaintingDataset
from tensorflow.keras.applications import InceptionV3
from util.transform import unnormalize, reverse_transform, img_tf, mask_tf, reverse_transform_list
import gc
torch.cuda.empty_cache()
gc.collect()

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

def evaluate_metric(snapshot, image_path, mask_path, batch_size = 4):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # load model
    model = PConvUNet().to(device)
    load_ckpt(snapshot, [('model', model)])

    data_eval = InpaintingDataset(image_path, mask_path, img_tf, mask_tf)

    iter_eval = iter(data.DataLoader(
        data_eval, batch_size=batch_size,
        sampler=InfiniteSampler(len(data_eval)),
        num_workers=8))

    model.eval()

    gt = []
    output = []

    for i in range(250):
        try:
            gt_i, mask_i, masked_img_i = [x.to(device) for x in next(iter_eval)]

            # generate output
            with torch.no_grad():
                output_i, _ = model(masked_img_i, mask_i)

            output_i = output_i.to(torch.device('cpu'))
            output_i = unnormalize(output_i)

            # concat
            gt_i = [x.permute(1, 2, 0).detach().cpu().numpy() for x in reverse_transform_list(gt_i)]
            output_i = [x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() for x in output_i]
            gt += gt_i
            output += output_i

        except StopIteration:
            print("Iterator is exhausted, stopping the loop.")
            break

    # calculate metric
    # prepare InceptionV3
    model_inception_v3 = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    print(calculate_fid(model_inception_v3, gt, output))



if __name__ == "__main__":
    snapshot = '../../../pretrain_checkpoint/checkpoint-Partial-Conv-10000.pth'
    data_path = '../../../dataset/celeba_hq/val/female'
    mask_path = '../../masks/mask_2'

    evaluate_metric(snapshot, data_path, mask_path)