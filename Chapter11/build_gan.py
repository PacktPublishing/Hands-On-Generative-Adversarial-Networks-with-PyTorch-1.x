# build_gan.py
# B11764 Chapter 11
# ==============================================

import os
import time
from datetime import datetime
import torch
from torch.optim.lr_scheduler import MultiStepLR
import utils
from model_3dgan import Generator as G
from model_3dgan import Discriminator as D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle


class Model(object):
    def __init__(self, name, device, data_loader, latent_dim, cube_len):
        self.name = name
        self.device = device
        self.data_loader = data_loader
        self.latent_dim = latent_dim
        self.cube_len = cube_len
        assert self.name == '3dgan'
        self.netG = G(self.latent_dim, self.cube_len)
        self.netG.to(self.device)
        self.netD = D(self.cube_len)
        self.netD.to(self.device)
        self.optim_G = None
        self.optim_D = None
        self.scheduler_D = None
        self.criterion = torch.nn.BCELoss()

    def create_optim(self, g_lr, d_lr, alpha=0.5, beta=0.5):
        self.optim_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=g_lr,
                                        betas=(alpha, beta))
        self.optim_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=d_lr,
                                        betas=(alpha, beta))
        self.scheduler_D = MultiStepLR(self.optim_D, milestones=[500, 1000])

    def train(self, epochs, d_loss_thresh, log_interval=100, export_interval=10, out_dir='', verbose=True):
        self.netG.train()
        self.netD.train()
        total_time = time.time()
        for epoch in range(epochs):
            batch_time = time.time()
            for batch_idx, data in enumerate(self.data_loader):
                data = data.to(self.device)

                batch_size = data.shape[0]
                real_label = torch.Tensor(batch_size).uniform_(
                    0.7, 1.2).to(self.device)
                fake_label = torch.Tensor(batch_size).uniform_(
                    0, 0.3).to(self.device)

                # Train D
                d_real = self.netD(data)
                d_real = d_real.squeeze()
                d_real_loss = self.criterion(d_real, real_label)

                latent = torch.Tensor(batch_size, self.latent_dim).normal_(
                    0, 0.33).to(self.device)
                fake = self.netG(latent)
                d_fake = self.netD(fake.detach())
                d_fake = d_fake.squeeze()
                d_fake_loss = self.criterion(d_fake, fake_label)

                d_loss = d_real_loss + d_fake_loss

                d_real_acc = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acc = torch.le(d_fake.squeeze(), 0.5).float()
                d_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc), 0))

                if d_acc <= d_loss_thresh:
                    self.netD.zero_grad()
                    d_loss.backward()
                    self.optim_D.step()

                # Train G
                latent = torch.Tensor(batch_size, self.latent_dim).normal_(
                    0, 0.33).to(self.device)
                fake = self.netG(latent)
                d_fake = self.netD(fake)
                d_fake = d_fake.squeeze()
                g_loss = self.criterion(d_fake, real_label)

                self.netD.zero_grad()
                self.netG.zero_grad()
                g_loss.backward()
                self.optim_G.step()

            if epoch % export_interval == 0:
                print(f'Working epoch {epoch} of {epochs}')
                samples = fake.cpu().data[:8].squeeze().numpy()
                # utils.save_voxels(samples, out_dir, epoch)
                save_voxels(samples, out_dir, epoch)
                # self.save_to(path=out_dir, name=self.name, verbose=False)
            self.scheduler_D.step()


def save_voxels(voxels, path, idx):
    from mpl_toolkits.mplot3d import Axes3D
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = matplotlib.gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)
    cnt = 1
    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        # ax = plt.subplot(gs[i], cnt, projection='3d')
        ax = fig.add_subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cnt += 1
    plt.savefig(path + '/{}.png'.format(str(idx)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(idx)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)
