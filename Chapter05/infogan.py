import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class Generator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Generator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.img_init_size = self.img_size // 4
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_init_shape = (128, self.img_init_size, self.img_init_size)
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.stem_linear = nn.Sequential(
            nn.Linear(latent_dim + classes + code_dim,
                      int(np.prod(self.img_init_shape)))
        )
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            *self._create_deconv_layer(128, 128, upsample=True),
            *self._create_deconv_layer(128, 64, upsample=True),
            *self._create_deconv_layer(64, self.channels, upsample=False, normalize=False),
            nn.Tanh()
        )

    def _create_deconv_layer(self, size_in, size_out, upsample=True, normalize=True):
        layers = []
        if upsample:
            layers.append(Upsample(scale_factor=2))
        layers.append(nn.Conv2d(size_in, size_out, 3, stride=1, padding=1))
        if normalize:
            layers.append(nn.BatchNorm2d(size_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels, code):
        z = torch.cat((noise, labels, code), -1)
        z_vec = self.stem_linear(z)
        z_img = z_vec.view(z_vec.shape[0], *self.img_init_shape)
        x = self.model(z_img)
        return x


class Discriminator(nn.Module):
    def __init__(self, classes, channels, img_size, latent_dim, code_dim):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.model = nn.Sequential(
            *self._create_conv_layer(self.channels, 16, True, False),
            *self._create_conv_layer(16, 32, True, True),
            *self._create_conv_layer(32, 64, True, True),
            *self._create_conv_layer(64, 128, True, True),
        )
        out_linear_dim = 128 * (self.img_size // 16) * (self.img_size // 16)
        self.adv_linear = nn.Linear(out_linear_dim, 1)
        self.class_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.classes)
        )
        self.code_linear = nn.Sequential(
            nn.Linear(out_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.code_dim)
        )
        self.adv_loss = torch.nn.MSELoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.style_loss = torch.nn.MSELoss()

    def _create_conv_layer(self, size_in, size_out, drop_out=True, normalize=True):
        layers = [nn.Conv2d(size_in, size_out, 3, 2, 1)]
        if drop_out:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.4))
        if normalize:
            layers.append(nn.BatchNorm2d(size_out, 0.8))
        return layers

    def forward(self, image):
        y_img = self.model(image)
        y_vec = y_img.view(y_img.shape[0], -1)
        y = self.adv_linear(y_vec)
        label = F.softmax(self.class_linear(y_vec), dim=1)
        code = self.code_linear(y_vec)
        return y, label, code
