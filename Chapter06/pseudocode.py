class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(64 * 8, 64 * 8, submodule=None, innermost=True)
        for i in range(8 - 5):
            unet_block = UnetSkipConnectionBlock(64 * 8, 64 * 8, submodule=unet_block, use_dropout=True)
        unet_block = UnetSkipConnectionBlock(64 * 4, 64 * 8, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(64 * 2, 64 * 4, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 64 * 2, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(3, 64, input_nc=3, submodule=unet_block, outermost=True)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    /* Innermost block */
    def __init__(self):
        down = [nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64 * 8, 64 * 8, kernel_size=4,
                          stride=2, padding=1, bias=False)]
        up = [nn.ReLU(inplace=True),
              nn.ConvTranspose2d(64 * 8, 64 * 8,
                                 kernel_size=4, stride=2,
                                 padding=1, bias=False),
              nn.BatchNorm2d(64 * 8)]
        model = down + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock(nn.Module):
    /* Other blocks */
    def __init__(self, out_channels, in_channels, submodule, use_dropout):
        down = [nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, in_channels, kernel_size=4,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels)]
        up = [nn.ReLU(inplace=True),
              nn.ConvTranspose2d(in_channels * 2, out_channels,
                                 kernel_size=4, stride=2,
                                 padding=1, bias=False),
              nn.BatchNorm2d(out_channels)]
        if use_dropout:
            model = down + [submodule] + up + [nn.Dropout(0.5)]
        else:
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock(nn.Module):
    /* Outermost block */
    def __init__(self):
        down = [nn.Conv2d(3, 64, kernel_size=4,
                          stride=2, padding=1, bias=False)]
        up = [nn.ReLU(inplace=True),
              nn.ConvTranspose2d(64 * 2, 3,
                                 kernel_size=4, stride=2,
                                 padding=1),
              nn.Tanh()]
        model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        sequence = [nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        channel_scale = 1
        channel_scale_prev = 1
        for n in range(1, n_layers):
            channel_scale_prev = channel_scale
            channel_scale = 2**n
            sequence += [
                nn.Conv2d(64 * channel_scale_prev, 64 * channel_scale, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64 * channel_scale),
                nn.LeakyReLU(0.2, True)
            ]
        channel_scale_prev = channel_scale
        sequence += [
            nn.Conv2d(64 * channel_scale_prev, 64 * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(64 * 8, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Pix2PixModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.netG = networks.define_G()
        self.netD = networks.define_D()

        self.criterionGAN = torch.nn.BCEWithLogitsLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * 100.0
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()