# main.py (partially provided)
# B11764 Chapter 11
# ========================================================================
# To launch program...
#  python main.py --model 3dgan --train True --data_dir DATA_DIRECTORY
# Be sure to set --data_dir to your data directory
# ========================================================================

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils

from build_gan import Model
from datasets import ShapeNetDataset

FLAGS = None


def main():
    now = datetime.now()
    dtstring = now.strftime('%X')
    print(f'Started at {dtstring}')
    # begin provided code...
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")

    print('Loading data...\n')
    dataset = ShapeNetDataset(FLAGS.data_dir, FLAGS.cube_len)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             FLAGS.batch_size,
                                             shuffle=True,
                                             num_workers=1,
                                             pin_memory=True)

    print('Creating model...\n')
    model = Model(FLAGS.model, device, dataloader,
                  FLAGS.latent_dim, FLAGS.cube_len)
    model.create_optim(FLAGS.g_lr, FLAGS.d_lr)

    # Train
    model.train(FLAGS.epochs, FLAGS.d_loss_thresh, FLAGS.log_interval,
                FLAGS.export_interval, FLAGS.out_dir, True)

    # end provided code
    now = datetime.now()
    dtstring = now.strftime('%X')
    print(f'Ended at {dtstring}')


if __name__ == '__main__':

    from utils import boolean_string
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 11')
    parser.add_argument('--model', type=str, default='3dGan',
                        help='enter `3dGan`.')
    parser.add_argument('--cube_len', type=int, default='32',
                        help='one of `cgan` and `infogan`.')
    parser.add_argument('--cuda', type=boolean_string,
                        default=True, help='enable CUDA.')
    parser.add_argument('--train', type=boolean_string,
                        default=True, help='train mode or eval mode.')
    parser.add_argument('--data_dir', type=str,
                        default='~/data', help='Directory for dataset.')
    parser.add_argument('--out_dir', type=str,
                        default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='size of batches')
    parser.add_argument('--g_lr', type=float, default=0.0002,
                        help='G learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002,
                        help='D learning rate')
    parser.add_argument('--d_loss_thresh', type=float, default=0.7,
                        help='D loss threshold')
    parser.add_argument('--latent_dim', type=int,
                        default=100, help='latent space dimension')
    parser.add_argument('--export_interval', type=int,
                        default=10, help='export interval')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--img_size', type=int,
                        default=64, help='size of images')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of image channels')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='interval between logging and image sampling')
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

    print(" " * 9 + "Args" + " " * 9 + "| " + "Type" +
          " | " + "Value")
    print("-" * 50)
    for arg in vars(FLAGS):
        arg_str = str(arg)
        var_str = str(getattr(FLAGS, arg))
        type_str = str(type(getattr(FLAGS, arg)).__name__)
        print(" " + arg_str + " " * (20-len(arg_str)) + "|" +
              " " + type_str + " " * (10-len(type_str)) + "|" +
              " " + var_str)
    main()
