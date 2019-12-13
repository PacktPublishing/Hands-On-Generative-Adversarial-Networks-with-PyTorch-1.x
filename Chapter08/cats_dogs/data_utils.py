import numpy as np
import torch
import torchvision.transforms as transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _transforms_catsdogs(args):
    CATS_DOGS_MEAN = [0.41649867, 0.45462068, 0.48786482]
    CATS_DOGS_STD = [0.22320463, 0.22301011, 0.22751113]

    # large_size = (int)(1.1 * args.img_size)

    train_transform = transforms.Compose([
        # transforms.Resize((large_size, large_size)),  # disabled due to speed issue
        # transforms.RandomCrop(args.img_size),         # disabled due to speed issue
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(CATS_DOGS_MEAN, CATS_DOGS_STD),  # diabled for better visualization
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(CATS_DOGS_MEAN, CATS_DOGS_STD),  # diabled for better visualization
        ])
    return train_transform, valid_transform


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target