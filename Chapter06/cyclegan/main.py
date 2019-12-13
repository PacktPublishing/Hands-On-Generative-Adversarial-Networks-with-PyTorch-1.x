import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from PIL import Image

import utils

from build_gan import Model
from datasets import ImageDataset

FLAGS = None

def main():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")

    if FLAGS.train:
        print('Loading data...\n')
        transform = [transforms.Resize(int(FLAGS.img_size*1.12), Image.BICUBIC),
                     transforms.RandomCrop((FLAGS.img_size, FLAGS.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        dataloader = DataLoader(ImageDataset(os.path.join(FLAGS.data_dir, FLAGS.dataset),
                                             transform=transform, unaligned=True, mode='train'),
                                batch_size=FLAGS.batch_size, shuffle=True, num_workers=2)
        test_dataloader = DataLoader(ImageDataset(os.path.join(FLAGS.data_dir, FLAGS.dataset),
                                                  transform=transform, unaligned=True, mode='test'),
                                     batch_size=FLAGS.test_batch_size, shuffle=True, num_workers=2)

        print('Creating model...\n')
        model = Model(FLAGS.model, device, dataloader, test_dataloader, FLAGS.channels, FLAGS.img_size, FLAGS.num_blocks)
        model.create_optim(FLAGS.lr)

        # Train
        model.train(FLAGS.epochs, FLAGS.log_interval, FLAGS.out_dir, True)

        model.save_to('')
    else:
        model = Model(FLAGS.model, device, None, test_dataloader, FLAGS.channels, FLAGS.img_size, FLAGS.num_blocks)
        model.load_from(FLAGS.out_dir)
        model.eval(mode=1, batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 6')
    parser.add_argument('--model', type=str, default='cyclegan', help='cyclegan')
    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=utils.boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='/media/john/HouseOfData/image_transfer', help='Directory for dataset.')
    parser.add_argument('--dataset', type=str, default='vangogh2photo', help='Dataset name.')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='size of batches in training')
    parser.add_argument('--test_batch_size', type=int, default=4, help='size of batches in inference')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='size of images')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--num_blocks', type=int, default=9, help='number of residual blocks')
    parser.add_argument('--log_interval', type=int, default=100, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    if FLAGS.train:
        utils.clear_folder(FLAGS.out_dir)

    log_file = os.path.join(FLAGS.out_dir, 'log.txt')
    print("Logging to {}\n".format(log_file))
    sys.stdout = utils.StdOut(log_file)

    print("PyTorch version: {}".format(torch.__version__))
    print("CUDA version: {}\n".format(torch.version.cuda))

    print(" " * 9 + "Args" + " " * 9 + "|    " + "Type" + \
          "    |    " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print("  " + arg_str + " " * (20-len(arg_str)) + "|" + \
              "  " + type_str + " " * (10-len(type_str)) + "|" + \
              "  " + var_str)

    main()
