#  model_3dgan.py
# B11764 Chapter 11
# ==============================================


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, cube_len, bias=False):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.cube_len = cube_len

        self.model = nn.Sequential(
            *self._create_layer(self.latent_dim, self.cube_len*8,
                                4, stride=2, padding=1, bias=bias, transposed=True),
            *self._create_layer(self.cube_len*8, self.cube_len*4,
                                4, stride=2, padding=1, bias=bias, transposed=True),
            *self._create_layer(self.cube_len*4, self.cube_len*2,
                                4, stride=2, padding=1, bias=bias, transposed=True),
            *self._create_layer(self.cube_len*2, self.cube_len,
                                4, stride=2, padding=1, bias=bias, transposed=True),
            *self._create_layer(self.cube_len, 1, 4, stride=2, padding=1, bias=bias, transposed=True, last_layer=True)
        )

    def _create_layer(self, size_in, size_out, kernel_size=4, stride=2, padding=1, bias=False, transposed=True, last_layer=False):
        layers = []
        if transposed:
            layers.append(nn.ConvTranspose3d(
                size_in, size_out, kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            layers.append(nn.Conv3d(size_in, size_out, kernel_size,
                                    stride=stride, padding=padding, bias=bias))
        if last_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.BatchNorm3d(size_out))
            layers.append(nn.ReLU(inplace=True))
        return layers

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1, 1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, cube_len, bias=False):
        super(Discriminator, self).__init__()
        self.cube_len = cube_len

        self.model = nn.Sequential(
            *self._create_layer(1, self.cube_len, 4, stride=2,
                                padding=1, bias=bias, transposed=False),
            *self._create_layer(self.cube_len, self.cube_len*2, 4,
                                stride=2, padding=1, bias=bias, transposed=False),
            *self._create_layer(self.cube_len*2, self.cube_len*4,
                                4, stride=2, padding=1, bias=bias, transposed=False),
            *self._create_layer(self.cube_len*4, self.cube_len*8,
                                4, stride=2, padding=1, bias=bias, transposed=False),
            *self._create_layer(self.cube_len*8, 1, 4, stride=2, padding=1, bias=bias, transposed=False, last_layer=True)
        )

    def _create_layer(self, size_in, size_out, kernel_size=4, stride=2, padding=1, bias=False, transposed=False, last_layer=False):
        layers = []
        if transposed:
            layers.append(nn.ConvTranspose3d(
                size_in, size_out, kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            layers.append(nn.Conv3d(size_in, size_out, kernel_size,
                                    stride=stride, padding=padding, bias=bias))
        if last_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.BatchNorm3d(size_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, x):
        x = x.view(-1, 1, self.cube_len, self.cube_len, self.cube_len)
        return self.model(x)
