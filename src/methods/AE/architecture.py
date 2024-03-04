'''
Same architecture from "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"
from Francesco Locatello & co.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from src.methods.baseline_model import Baseline
from torch import nn


class Encoder(nn.Module):

    def __init__(self, latent_dim, n_channel, **kwargs):
        super(Encoder, self).__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(n_channel, 16, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(in_features=4096, out_features=latent_dim, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv1_2(x))
        x = self.pool(x)
        x = self.flatten(x)

        x = self.linear(x)

        return x


class Decoder(nn.Module):
    def __init__(self, data_shape, latent_dim, n_channel, **kwargs):
        super(Decoder, self).__init__()
        self.data_shape= np.array([-1] + [data_shape[-1]] + data_shape[:-1])

        # Decoder
        self.linear1 = nn.Linear(in_features=latent_dim, out_features=4096, bias=True)

        self.t_conv2_2 = nn.ConvTranspose2d(16, 16, 2, stride=2)

        self.final_t_conv = nn.ConvTranspose2d(16, n_channel, 2, stride=2)

    def forward(self, x):

        x = self.linear1(x)

        x = x.view(-1, 16, 16, 16)  # reshape (batch_size x channels x 4 x 4)
        x = F.leaky_relu(self.t_conv2_2(x))

        x = self.final_t_conv(x, output_size =self.data_shape)  # reconstruction

        x = torch.sigmoid(x)
        return x


class AE(Baseline):

    def __init__(self, data_shape, latent_dim, n_channel, **kwargs):
        super(AE, self).__init__(**kwargs)

        self.encoder = Encoder(latent_dim, n_channel)
        self.decoder = Decoder(data_shape,  latent_dim, n_channel)

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        # save latent dimension
        self.latent_dim = latent_dim

        self.device = None



    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x

    def encode(self, x):
        ''' Return representation given a sample as only a point in the latent space'''
        c = self.encoder.forward(x)
        return {"mean" : c}


    def sample(self, num=25):
        # mean and variance of latent code (better to estimate them with a test_unsupervised set)
        mean = 0.
        std = 1.

        # sample latent vectors from the normal distribution
        latents = torch.randn(num, self.latent_dim) * std + mean
        imgs = self.decoder.forward(latents.to(self.device))
        return imgs


    def decode(self, code):
        c = self.decoder.forward(code.to(self.device))
        return c.cpu().detach().numpy()


    def compute_loss(self, y):
        loss, y_hat = super(AE, self).compute_loss(y)
        loss = {"loss": loss}
        return loss , y_hat # * torch.prod(torch.tensor(y_hat.size()))

    def save_state(self, path):
        ''' Save model state, including criterion and optimiizer '''
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, path)

    def load_state(self, path):
        ''' Load model state, including criterion and optimiizer '''

        # load the model checkpoint
        checkpoint = torch.load(path)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


