import itertools
import os
import time

from datetime import datetime

import numpy as np
import torch
import torchvision.utils as vutils

import utils

from cgan import Generator as cganG
from cgan import Discriminator as cganD
from infogan import Generator as infoganG
from infogan import Discriminator as infoganD


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
                 classes,
                 channels,
                 img_size,
                 latent_dim,
                 style_dim=2):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        if self.name == 'cgan':
            self.netG = cganG(self.classes, self.channels, self.img_size, self.latent_dim)
        elif self.name == 'infogan':
            self.netG = infoganG(self.classes, self.channels, self.img_size, self.latent_dim, self.style_dim)
            self.netG.apply(_weights_init)
        self.netG.to(self.device)
        if self.name == 'cgan':
            self.netD = cganD(self.classes, self.channels, self.img_size, self.latent_dim)
        elif self.name == 'infogan':
            self.netD = infoganD(self.classes, self.channels, self.img_size, self.latent_dim, self.style_dim)
            self.netD.apply(_weights_init)
        self.netD.to(self.device)
        self.infogan = self.name == 'infogan'
        self.optim_G = None
        self.optim_D = None
        self.optim_info = None

    @property
    def generator(self):
        return self.netG

    @property
    def discriminator(self):
        return self.netD

    def create_optim(self, lr, alpha=0.5, beta=0.999):
        self.optim_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        self.netG.parameters()),
                                        lr=lr,
                                        betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        self.netD.parameters()),
                                        lr=lr,
                                        betas=(alpha, beta))
        if self.infogan:
            self.optim_info = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netD.parameters()),
                                               lr=lr, betas=(alpha, beta))

    def _to_onehot(self, var, dim):
        res = torch.zeros((var.shape[0], dim), device=self.device)
        res[range(var.shape[0]), var] = 1.
        return res

    def train(self,
              epochs,
              log_interval=100,
              out_dir='',
              verbose=True):
        self.netG.train()
        self.netD.train()
        viz_z = torch.zeros((self.data_loader.batch_size, self.latent_dim), device=self.device)
        viz_noise = torch.randn(self.data_loader.batch_size, self.latent_dim, device=self.device)
        nrows = self.data_loader.batch_size // 8
        viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(self.device)
        viz_onehot = self._to_onehot(viz_label, dim=self.classes)
        viz_style = torch.zeros((self.data_loader.batch_size, self.style_dim), device=self.device)
        total_time = time.time()
        for epoch in range(epochs):
            batch_time = time.time()
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1., device=self.device)
                fake_label = torch.full((batch_size, 1), 0., device=self.device)

                # Train G
                self.netG.zero_grad()
                z_noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                if self.infogan:
                    labels_onehot = self._to_onehot(x_fake_labels, dim=self.classes)
                    z_style = torch.zeros((batch_size, self.style_dim), device=self.device).normal_()

                    x_fake = self.netG(z_noise, labels_onehot, z_style)
                    y_fake_g, _, _ = self.netD(x_fake)
                    g_loss = self.netD.adv_loss(y_fake_g, real_label)
                else:
                    x_fake = self.netG(z_noise, x_fake_labels)
                    y_fake_g = self.netD(x_fake, x_fake_labels)
                    g_loss = self.netD.loss(y_fake_g, real_label)
                g_loss.backward()
                self.optim_G.step()

                # Train D
                self.netD.zero_grad()
                if self.infogan:
                    y_real, _, _ = self.netD(data)
                    d_real_loss = self.netD.adv_loss(y_real, real_label)

                    y_fake_d, _, _ = self.netD(x_fake.detach())
                    d_fake_loss = self.netD.adv_loss(y_fake_d, fake_label)
                else:
                    y_real = self.netD(data, target)
                    d_real_loss = self.netD.loss(y_real, real_label)

                    y_fake_d = self.netD(x_fake.detach(), x_fake_labels)
                    d_fake_loss = self.netD.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optim_D.step()

                if self.infogan:
                    # Update mutual information
                    self.optim_info.zero_grad()
                    z_noise.normal_()
                    x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                    labels_onehot = self._to_onehot(x_fake_labels, dim=self.classes)
                    z_style.normal_()
                    x_fake = self.netG(z_noise, labels_onehot, z_style)
                    _, label_fake, style_fake = self.netD(x_fake)
                    info_loss = self.netD.class_loss(label_fake, x_fake_labels) +\
                                self.netD.style_loss(style_fake, z_style)
                    info_loss.backward()
                    self.optim_info.step()

                if verbose and batch_idx % log_interval == 0 and batch_idx > 0:
                    if self.infogan:
                        print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} loss_I: {:.4f} time: {:.2f}'.format(
                              epoch, batch_idx, len(self.data_loader),
                              d_loss.mean().item(),
                              g_loss.mean().item(),
                              info_loss.mean().item(),
                              time.time() - batch_time))
                    else:
                        print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                              epoch, batch_idx, len(self.data_loader),
                              d_loss.mean().item(),
                              g_loss.mean().item(),
                              time.time() - batch_time))
                    vutils.save_image(data, os.path.join(out_dir, 'real_samples.png'), normalize=True)
                    with torch.no_grad():
                        if self.infogan:
                            viz_sample = self.netG(viz_noise, viz_onehot, viz_style)
                            vutils.save_image(viz_sample, os.path.join(out_dir, 'fake_samples_{}.png'.format(epoch)), nrow=8, normalize=True)
                            # zeros = np.zeros((self.data_loader.batch_size, 1))
                            # c_varied = np.repeat(np.linspace(-1, 1, self.classes)[:, np.newaxis], self.classes, 0)
                            # c2 = torch.FloatTensor(np.concatenate((c_varied, zeros), -1)).to(self.device)
                            # c3 = torch.FloatTensor(np.concatenate((zeros, c_varied), -1)).to(self.device)
                            # sample1 = self.netG(viz_z, viz_onehot, c2)
                            # sample2 = self.netG(viz_z, viz_onehot, c3)
                            # vutils.save_image(sample1, os.path.join(out_dir, 'c2/fake_samples_{}.png'.format(epoch)), nrow=self.classes, normalize=True)
                            # vutils.save_image(sample2, os.path.join(out_dir, 'c3/fake_samples_{}.png'.format(epoch)), nrow=self.classes, normalize=True)
                        else:
                            viz_sample = self.netG(viz_noise, viz_label)
                            vutils.save_image(viz_sample, os.path.join(out_dir, 'fake_samples_{}.png'.format(epoch)), nrow=8, normalize=True)
                    batch_time = time.time()

            self.save_to(path=out_dir, name=self.name, verbose=False)
        if verbose:
            print('Total train time: {:.2f}'.format(time.time() - total_time))

    def eval(self,
             mode=None,
             batch_size=None):
        self.netG.eval()
        self.netD.eval()
        if batch_size is None:
            batch_size = self.data_loader.batch_size
        nrows = batch_size // 8
        viz_labels = np.array([num for _ in range(nrows) for num in range(8)])
        viz_labels = torch.LongTensor(viz_labels).to(self.device)

        with torch.no_grad():
            if self.infogan:
                viz_tensor = torch.randn(batch_size, self.latent_dim, device=self.device)
                labels_onehot = self._to_onehot(viz_labels, dim=self.classes)
                z_style = torch.zeros((batch_size, self.style_dim), device=self.device)
                if mode is not None:
                    for i in range(batch_size):
                        z_style[i, mode] = 4. * i / batch_size - 2.
                viz_sample = self.netG(viz_tensor, labels_onehot, z_style)
            else:
                viz_tensor = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
                viz_sample = self.netG(viz_tensor, viz_labels)
            viz_vector = utils.to_np(viz_tensor).reshape(batch_size, self.latent_dim)
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            np.savetxt('vec_{}.txt'.format(cur_time), viz_vector)
            vutils.save_image(viz_sample, 'img_{}.png'.format(cur_time), nrow=8, normalize=True)

    def save_to(self,
                path='',
                name=None,
                verbose=True):
        if name is None:
            name = self.name
        if verbose:
            print('\nSaving models to {}_G.pt and {}_D.pt ...'.format(name, name))
        torch.save(self.netG.state_dict(), os.path.join(path, '{}_G.pt'.format(name)))
        torch.save(self.netD.state_dict(), os.path.join(path, '{}_D.pt'.format(name)))

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
        else:
            self.netG.load_state_dict(ckpt_G, strict=True)
        ckpt_D = torch.load(os.path.join(path, '{}_D.pt'.format(name)))
        if isinstance(ckpt_D, dict) and 'state_dict' in ckpt_D:
            self.netD.load_state_dict(ckpt_D['state_dict'], strict=True)
        else:
            self.netD.load_state_dict(ckpt_D, strict=True)
