import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from libs.PartialConv.util import opt
import numpy as np

from libs.PartialConv.model.loss import InpaintingLoss
from libs.PartialConv.model.model import VGG16FeatureExtractor, PConvUNet
from libs.PartialConv.util.io import load_ckpt, save_ckpt
from libs.PartialConv.util.places2 import Places2

IMAGE_SIZE = 512
size = (IMAGE_SIZE, IMAGE_SIZE)
DATASET_DIR = 'celeba_modify'
MASK_DIR = 'masks'

save_dir = 'snapshots/default'
log_dir ='logs/default'
lr = 2e-4
lr_finetune = 5e-5
num_epochs = 100
max_iter=10000
batch_size=4
n_threads=8
save_model_interval=50000
vis_interval=5000
log_interval=10

resume=str
finetune='store_true'

img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])

mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataset
    dataset_train = Places2(args.root, args.mask_root, img_tf, mask_tf, 'train')
    dataset_val = Places2(args.root, args.mask_root, img_tf, mask_tf, 'val')

    # get iteratior of training dataset
    iterator_train = iter(data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=InfiniteSampler(len(dataset_train)),
        num_workers=args.n_threads))

    # get model
    model = PConvUNet().to(device)
    start_iter = 0
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    # check if finetune
    if args.finetune:
        lr = args.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = args.lr

    # check if resume
    if args.resume:
        start_iter = load_ckpt(
            args.resume, [('model', model)], [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    for epoch in range(num_epochs):
        loss = 0.0
        for i in tqdm(range(start_iter, args.max_iter)):
            model.train()

            image, mask, gt = [x.to(device) for x in next(iterator_train)]
            output, _ = model(image, mask)
            loss_dict = criterion(image, mask, output, gt)

            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value
                if (i + 1) % args.log_interval == 0:
                    writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
                save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                          [('model', model)], [('optimizer', optimizer)], i + 1)

            if (i + 1) % args.vis_interval == 0:
                model.eval()
                evaluate(model, dataset_val, device,
                         '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

        print(f"Epoch {epoch + 1}/{num_epochs} . loss: {loss}")


if __name__ == "__main__":
    train()