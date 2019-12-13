import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.utils as vutils

import utils
from advGAN import AdvGAN_Attack
from data_utils import data_prefetcher, _transforms_catsdogs
from model_ensemble import transfer_init, ModelEnsemble


FLAGS = None

def main():
    device = torch.device("cuda:0" if FLAGS.cuda else "cpu")

    print('Loading data...\n')
    train_transform, _ = _transforms_catsdogs(FLAGS)
    train_data = dset.ImageFolder(root=FLAGS.data_dir, transform=train_transform)
    assert train_data

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(FLAGS.data_split * num_train))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=FLAGS.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=2)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=FLAGS.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=2)

    if FLAGS.train_single:
        print('Transfer training model {}...\n'.format(FLAGS.model))
        model = torch.hub.load('pytorch/vision', FLAGS.model, pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model, param_to_train = transfer_init(model, FLAGS.model, FLAGS.classes)
        model.to(device)

        optimizer = torch.optim.SGD(
            param_to_train, FLAGS.lr,
            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        # Train
        best_acc = 0.0
        for epoch in range(25):
            model.train()
            scheduler.step()
            print('Epoch {}, lr: {}'.format(epoch, scheduler.get_lr()[0]))
            prefetcher = data_prefetcher(train_loader)
            data, target = prefetcher.next()
            batch_idx = 0
            while data is not None:
                optimizer.zero_grad()
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                correct = pred.eq(target.view_as(pred)).sum().item()
                if batch_idx % FLAGS.log_interval == 0:
                    print('[{}/{}]\tloss: {:.4f}\tbatch accuracy: {:.4f}%'.format(
                        batch_idx * FLAGS.batch_size, num_train,
                        loss.item(), 100 * correct / data.size(0)))
                data, target = prefetcher.next()
                batch_idx += 1
            # Eval
            model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                valid_prefetcher = data_prefetcher(valid_loader)
                data, target = valid_prefetcher.next()
                while data is not None:
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.max(1, keepdim=True)[1]
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
                    data, target = valid_prefetcher.next()
            test_loss /= len(valid_loader)
            test_correct = 100 * test_correct / (1-FLAGS.data_split) / num_train
            print('Eval loss: {:.4f}, accuracy: {:.4f}'.format(
                test_loss, test_correct))
            if (test_correct > best_acc):
                best_acc = test_correct
                torch.save(model, os.path.join(FLAGS.model_dir, '{}.pth'.format(FLAGS.model)))
        print('Best test accuracy for model {}: {:.4f}'.format(FLAGS.model, best_acc))
    elif FLAGS.train_ensemble:
        print('Loading model...\n')
        model_names = ['mobilenet_v2', 'resnet18', 'densenet121',
                       'googlenet', 'resnext50_32x4d']
        model = ModelEnsemble(model_names, FLAGS.classes, FLAGS.model_dir)
        model.to(device)

        optimizer = torch.optim.SGD(
            model.vote_layer.parameters(), FLAGS.lr,
            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        criterion = nn.CrossEntropyLoss()

        # Train
        print('Training ensemble model...\n')
        # model = torch.load(os.path.join(FLAGS.model_dir, 'ensemble.pth'))
        for epoch in range(2):
            model.train()
            scheduler.step()
            print('Epoch {}, lr: {}'.format(epoch, scheduler.get_lr()[0]))
            prefetcher = data_prefetcher(train_loader)
            data, target = prefetcher.next()
            batch_idx = 0
            while data is not None:
                optimizer.zero_grad()
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                correct = pred.eq(target.view_as(pred)).sum().item()
                if batch_idx % FLAGS.log_interval == 0:
                    print('[{}/{}]\tloss: {:.4f}\tbatch accuracy: {:.4f}%'.format(
                        batch_idx * FLAGS.batch_size, num_train,
                        loss.item(), 100 * correct / data.size(0)))
                data, target = prefetcher.next()
                batch_idx += 1
            # Eval
            model.eval()
            test_loss = 0
            test_correct = 0
            valid_prefetcher = data_prefetcher(valid_loader)
            data, target = valid_prefetcher.next()
            while data is not None:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                data, target = valid_prefetcher.next()

            test_loss /= len(valid_loader)
            print('Eval loss: {:.4f}, accuracy: {:.4f}'.format(
                test_loss, 100 * test_correct / (1-FLAGS.data_split) / num_train))
        torch.save(model, os.path.join(FLAGS.model_dir, 'ensemble.pth'))

        # Adversarial attack
        print('Training GAN for adversarial attack...\n')
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=16,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=2)

        model.eval()
        advGAN = AdvGAN_Attack(device, model, FLAGS.classes,
                               FLAGS.channels, 0, 1, FLAGS.model_dir)
        # advGAN.netG = torch.load(os.path.join(FLAGS.model_dir, 'netG_epoch_{}.pth'.format(FLAGS.pretrained_epoch)))

        advGAN.train(train_loader, FLAGS.epochs)

        print('Attacking ensemble model...\n')
        test_loss = 0
        test_correct = 0
        adv_examples = []
        # enough = False
        with torch.no_grad():
            valid_prefetcher = data_prefetcher(valid_loader)
            data, target = valid_prefetcher.next()
            while data is not None:
                # for i in range(64):
                #     adv_ex = data[i].squeeze().detach().cpu().numpy()
                #     adv_examples.append((0, 0, adv_ex))
                # break
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1]
                init_pred = init_pred.view_as(target)

                perturbed_data = advGAN.adv_example(data)
                output = model(perturbed_data)
                test_loss += criterion(output, target).item()
                final_pred = output.max(1, keepdim=True)[1]
                final_pred = final_pred.view_as(target)
                test_correct += final_pred.eq(target).sum().item()
                if len(adv_examples) < 64 and not (final_pred == target).all():
                    indices = torch.ne(final_pred.ne(target), init_pred.ne(target)).nonzero()
                    for i in range(indices.shape[0]):
                        adv_ex = perturbed_data[indices[i]].squeeze().detach().cpu().numpy()
                        adv_examples.append((init_pred[indices[i]].item(), final_pred[indices[i]].item(), adv_ex))
                        # adv_ex = perturbed_data[indices[i]].squeeze()
                        # adv_examples.append(adv_ex)
                        if (len(adv_examples) >= 64):
                            # enough = True
                            break
                # if enough:
                #     break
                data, target = valid_prefetcher.next()
        test_loss /= len(valid_loader)
        print('Eval loss: {:.4f}, accuracy: {:.4f}'.format(
            test_loss, 100 * test_correct / (1-FLAGS.data_split) / num_train))

        # show 64 results
        if True:
            cnt = 0
            plt.figure(figsize=(8,10))
            for i in range(8):
                for j in range(8):
                    cnt += 1
                    plt.subplot(8, 8, cnt)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    orig, adv, ex = adv_examples[i*8+j]
                    ex = np.transpose(ex, (1, 2, 0))
                    plt.title("{} -> {}".format(orig, adv))
                    plt.imshow(ex)
            plt.tight_layout()
            plt.show()
        else:
            viz_sample = torch.stack(adv_examples, dim=0)
            print(viz_sample.shape)
            vutils.save_image(viz_sample, 'adv_examples.png', nrow=8, normalize=True)


if __name__ == '__main__':
    from utils import boolean_string
    legal_models = ['resnet18', 'resnet34', 'mobilenet_v2', 'shufflenet_v2_x1_0',
                    'squeezenet1_1', 'densenet121', 'googlenet', 'resnext50_32x4d',
                    'vgg11']
    parser = argparse.ArgumentParser(description='Hands-On GANs - Chapter 8')
    parser.add_argument('--model', type=str, default='resnet18', help='one of {}'.format(legal_models))
    parser.add_argument('--cuda', type=boolean_string, default=True, help='enable CUDA.')
    parser.add_argument('--train_single', type=boolean_string, default=True, help='train single model.')
    parser.add_argument('--train_ensemble', type=boolean_string, default=True, help='train final model.')
    parser.add_argument('--model_dir', type=str, default='models', help='directory for trained models')
    parser.add_argument('--data_dir', type=str, default='/media/john/FastData/cats-dogs-kaggle/train', help='Directory for dataset.')
    parser.add_argument('--data_split', type=float, default=0.8, help='split ratio for train and val data')
    parser.add_argument('--cutout', type=boolean_string, default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=64, help='number of epochs')
    parser.add_argument('--out_dir', type=str, default='output', help='Directory for output.')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--classes', type=int, default=2, help='number of classes')
    parser.add_argument('--img_size', type=int, default=224, help='size of images')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--log_interval', type=int, default=50, help='interval between logging and image sampling')
    parser.add_argument('--pretrained_epoch', type=int, default=60, help='epoch number of pretrained generator')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    FLAGS = parser.parse_args()
    FLAGS.cuda = FLAGS.cuda and torch.cuda.is_available()
    assert FLAGS.model in legal_models

    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        if FLAGS.cuda:
            torch.cuda.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    cudnn.benchmark = True

    try:
        import accimage
        torchvision.set_image_backend('accimage')
        print('Image loader backend: accimage')
    except:
        print('Image loader backend: PIL')

    if FLAGS.train_single:
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
