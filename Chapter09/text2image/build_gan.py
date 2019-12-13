import itertools
import os
import time

from datetime import datetime

import numpy as np
import torch
import torchvision.utils as vutils

import utils

from gan import Generator as netG
from gan import Discriminator as netD


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Model(object):
    def __init__(self,
                 name,
                 device,
                 data_loader,
                 channels,
                 l1_coef,
                 l2_coef):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.channels = channels
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.netG = netG(self.channels)
        self.netG.apply(_weights_init)
        self.netG.to(self.device)
        self.netD = netD(self.channels)
        self.netD.apply(_weights_init)
        self.netD.to(self.device)
        self.optim_G = None
        self.optim_D = None
        self.loss_adv = torch.nn.BCELoss()
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_l2 = torch.nn.MSELoss()

    @property
    def generator(self):
        return self.netG

    @property
    def discriminator(self):
        return self.netD

    def create_optim(self, lr, alpha=0.5, beta=0.999):
        self.optim_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=lr, betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=lr, betas=(alpha, beta))

    def train(self,
              epochs,
              log_interval=100,
              out_dir='',
              verbose=True):
        self.netG.train()
        self.netD.train()
        total_time = time.time()
        for epoch in range(epochs):
            batch_time = time.time()
            for batch_idx, data in enumerate(self.data_loader):
                image = data['right_images'].to(self.device)
                embed = data['right_embed'].to(self.device)

                real_label = torch.ones((image.shape[0]), device=self.device)
                fake_label = torch.zeros((image.shape[0]), device=self.device)

                # Train D
                self.optim_D.zero_grad()

                out_real, _ = self.netD(image, embed)
                loss_d_real = self.loss_adv(out_real, real_label)

                noise = torch.randn((image.shape[0], 100, 1, 1), device=self.device)
                image_fake = self.netG(noise, embed)
                out_fake, _ = self.netD(image_fake, embed)
                loss_d_fake = self.loss_adv(out_fake, fake_label)

                d_loss = loss_d_real + loss_d_fake
                d_loss.backward()
                self.optim_D.step()

                # Train G
                self.optim_G.zero_grad()
                noise = torch.randn((image.shape[0], 100, 1, 1), device=self.device)
                image_fake = self.netG(noise, embed)
                out_fake, act_fake = self.netD(image_fake, embed)
                _, act_real = self.netD(image, embed)

                l1_loss = self.loss_l1(torch.mean(act_fake, 0), torch.mean(act_real, 0).detach())
                g_loss = self.loss_adv(out_fake, real_label) + \
                    self.l1_coef * l1_loss + \
                    self.l2_coef * self.loss_l2(image_fake, image)
                g_loss.backward()
                self.optim_G.step()

                if verbose and batch_idx % log_interval == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                          epoch, batch_idx, len(self.data_loader),
                          d_loss.mean().item(),
                          g_loss.mean().item(),
                          time.time() - batch_time))
                    with torch.no_grad():
                        viz_sample = torch.cat((image[:32], image_fake[:32]), 0)
                        vutils.save_image(viz_sample,
                                          os.path.join(out_dir, 'samples_{}_{}.png'.format(epoch, batch_idx)),
                                          nrow=8,
                                          normalize=True)
                    batch_time = time.time()

            self.save_to(path=out_dir, name=self.name, verbose=False)
        if verbose:
            print('Total train time: {:.2f}'.format(time.time() - total_time))

    def eval(self,
             batch_size=None):
        self.netG.eval()
        self.netD.eval()
        if batch_size is None:
            batch_size = self.data_loader.batch_size

        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader):
                image = data['right_images'].to(self.device)[:batch_size]
                embed = data['right_embed'].to(self.device)[:batch_size]
                text = data['txt'][:batch_size]
                noise = torch.randn((image.shape[0], 100, 1, 1), device=self.device)
                viz_sample = self.netG(noise, embed)
                vutils.save_image(viz_sample,
                                  'img_{}.png'.format(batch_idx),
                                  nrow=batch_size//8,
                                  normalize=True)
                for t in text:
                    print(t)
                break

    def save_to(self,
                path='',
                name=None,
                verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nSaving models to {}_G.pt and {}_D.pt ...'.format(name, name))
        torch.save(self.netG, os.path.join(path, '{}_G.pt'.format(name)))
        torch.save(self.netD, os.path.join(path, '{}_D.pt'.format(name)))

    def load_from(self,
                  path='',
                  name=None,
                  verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nLoading models from {}_G.pt and {}_D.pt ...'.format(name, name))
        ckpt_G = torch.load(os.path.join(path, '{}_G.pt'.format(name)))
        if isinstance(ckpt_G, dict) and 'state_dict' in ckpt_G:
            self.netG.load_state_dict(ckpt_G['state_dict'], strict=True)
        elif isinstance(ckpt_G, torch.nn.Module):
            self.netG = ckpt_G
        else:
            self.netG.load_state_dict(ckpt_G, strict=True)
        ckpt_D = torch.load(os.path.join(path, '{}_D.pt'.format(name)))
        if isinstance(ckpt_D, dict) and 'state_dict' in ckpt_D:
            self.netD.load_state_dict(ckpt_D['state_dict'], strict=True)
        elif isinstance(ckpt_D, torch.nn.Module):
            self.netD = ckpt_D
        else:
            self.netD.load_state_dict(ckpt_D, strict=True)
