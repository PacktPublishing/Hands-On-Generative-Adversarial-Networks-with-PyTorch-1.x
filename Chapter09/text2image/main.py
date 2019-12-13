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
from txt2image_dataset import Text2ImageDataset

FLAGS = None

def main():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")

    print('Loading data...\n')
    dataloader = DataLoader(Text2ImageDataset(os.path.join(FLAGS.data_dir, '{}.hdf5'.format(FLAGS.dataset)), split=0),
                            batch_size=FLAGS.batch_size, shuffle=True, num_workers=8)

    print('Creating model...\n')
    model = Model(FLAGS.model, device, dataloader, FLAGS.channels, FLAGS.l1_coef, FLAGS.l2_coef)

    if FLAGS.train:
        model.create_optim(FLAGS.lr)

        print('Training...\n')
        model.train(FLAGS.epochs, FLAGS.log_interval, FLAGS.out_dir, True)

        model.save_to('')
    else:
        model.load_from('')

        print('Evaluating...\n')
        model.eval(batch_size=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 9')
    parser.add_argument('--model', type=str, default='text2image', help='text2image')
    parser.add_argument('--cuda', type=utils.boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train', type=utils.boolean_string, default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str, default='/media/john/DataAsgard/text2image/birds', help='Directory for dataset.')
    parser.add_argument('--dataset', type=str, default='birds', help='Dataset name.')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='size of batches in training')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--l1_coef', type=float, default=50, help='l1 coefficient')
    parser.add_argument('--l2_coef', type=float, default=100, help='l2 coefficient')
    parser.add_argument('--log_interval', type=int, default=20, help='interval between logging and image sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    utils.create_folder(FLAGS.out_dir)
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
