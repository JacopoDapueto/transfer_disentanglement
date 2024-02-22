

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from src.methods.shared.utils.ssim import multiscale_structural_similarity_index_measure as mssim
from src.methods.shared.utils.ssim import structural_similarity_index_measure as ssim
from torch import nn

from src.methods.AE.architecture import AE


class Encoder(nn.Module):

    def __init__(self, n_filters, input_shape, latent_dim, n_channel, **kwargs):
        super(Encoder, self).__init__()

        self.d = n_filters

        self.layer_count = 4

        # Encoder
        #self.conv1_1 = nn.Conv2d(n_channel, 32, 4, stride=2, padding=0)

        #self.conv1_2 = nn.Conv2d(32, 32, 4, stride=2, padding=0)
        #self.conv2_1 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        #self.conv2_2 = nn.Conv2d(64, 64, 4, stride=2, padding=0)
        #self.flatten = nn.Flatten()

        #self.linear = nn.Linear(in_features=2304, out_features=latent_dim, bias=True) # 256 --> 64, 2304 --> 128

        inputs = n_channel
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, self.d, 4, stride= 2, padding=1))
            #print(inputs, self.d)
            #setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = self.d #* mul
            if (i+1)%2==0:
                self.d *=2

        self.linear_dim = 4 if input_shape == 64 else 8 if input_shape == 128 else 2
        self.linear = nn.Linear(in_features=inputs * self.linear_dim * self.linear_dim, out_features=latent_dim, bias=True)

        self.d_max = inputs


    def forward(self, x):
        #x = F.leaky_relu(self.conv1_1(x))
        #print(x.size())
        #x = F.leaky_relu(self.conv1_2(x))
        #print(x.size())
        #x = F.leaky_relu(self.conv2_1(x))
        #x = F.leaky_relu(self.conv2_2(x))
        #print(x.size())
        #x = self.flatten(x)
        #print(x.size())

        #x = self.linear(x)

        for i in range(self.layer_count):
            x = F.leaky_relu(getattr(self, "conv%d" % (i + 1))(x))
            #print(x.size())

        #print(self.d_max * 4 *4)
        #print(x.size())

        x = x.view(x.size(0), self.d_max * self.linear_dim * self.linear_dim)
        #print(x.size)

        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_filters, inputs, data_shape, latent_dim, n_channel, **kwargs):
        super(Decoder, self).__init__()


        self.inputs = inputs
        self.d_max = inputs
        self.layer_count = 4
        self.d = n_filters * (self.layer_count // 2)

        self.data_shape= np.array([-1] + [data_shape[-1]] + data_shape[:-1])



        self.linear_dim = 4 if data_shape[0] == 64 else 8 if data_shape[0] == 128 else 2

        self.linear = nn.Linear(in_features=latent_dim, out_features=inputs * self.linear_dim * self.linear_dim, bias=True)


        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d, 4, 2, 1))
            inputs = self.d
            if (i + 1) % 2 == 0:
                self.d //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, n_channel, 4, 2, 1))

    def forward(self, x):



        x = self.linear(x)

        x = x.view(x.size(0), self.d_max, self.linear_dim, self.linear_dim)
        for i in range(1, self.layer_count):
            x = F.leaky_relu((getattr(self, "deconv%d" % (i + 1))(x)))

        x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))

        return (x + 1.)/2.


class VAE(AE):

    def __init__(self, n_filters, decoder_distribution,  beta, data_shape, latent_dim, n_channel, criterion, **kwargs):
        super(VAE, self).__init__(data_shape, latent_dim, n_channel,criterion, **kwargs)

        self.encoder = Encoder(n_filters, data_shape[0], latent_dim, n_channel)
        self.decoder = Decoder(n_filters, self.encoder.d_max, data_shape, latent_dim, n_channel)

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)


        # distribution parameters
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.beta = beta
        self.decoder_distribution = decoder_distribution # "gaussian" # "bernoulli"
        self.subtract_true_image_entropy = False
        self.data_shape = data_shape



    def forward(self, x):

        x = self.encoder.forward(x)
        mu, log_var = self.fc_mu(x), self.fc_var(x)

        z = self.z_sample(mu, log_var)

        x = self.decoder.forward(z)

        return x

    def z_sample(self, mu, log_var):
        # sample z from q
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def encode(self, x):
        ''' Return representation given a sample as only a point in the latent space'''
        x = self.encoder.forward(x)

        mu, log_var = self.fc_mu(x), self.fc_var(x)

        z = self.z_sample(mu, log_var)

        return {"mean" : mu, "std": torch.exp(0.5 * log_var ), "sampled":z}

    def sample(self, num=25):
        # mean and variance of latent code (better to estimate them with a test set)
        mean = 0.
        std = 1.

        # sample latent vectors from the normal distribution
        latents = torch.randn(num, self.latent_dim) * std + mean
        imgs = self.decode(latents) #self.decoder.forward(latents.to(self.device))
        return imgs

    def decode(self, code):
        c = self.decoder.forward(code.to(self.device))

        return c

    def compute_loss(self, y):
        x_encoded = self.encoder.forward(y)
        mu, log_var = self.fc_mu(x_encoded ), self.fc_var(x_encoded )

        z = self.z_sample(mu, log_var)

        y_hat = self.decoder.forward(z)

        # reconstruction loss
        recon_loss = self.reconstruction(y_hat, self.log_scale, y).mean()

        # kl
        kl = self.kl_divergence( mu, log_var).mean()

        elbo = recon_loss - self.beta * kl
        loss = {"kl": kl, "reconstruction": recon_loss, "loss":-elbo, "elbo": recon_loss -  kl}
        return  loss, y_hat  # * torch.prod(torch.tensor(y_hat.size()))

    def kl_divergence(self, mu, log_var):

        return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)

    def reconstruction(self, x_hat, logscale, x):

        if self.decoder_distribution == "mssim":
            return -(1 - mssim(x_hat, x, kernel_size=7,  reduction=None, data_range=1.0, betas=(0.0448, 0.2856, 0.3001))) * (self.data_shape[0] * self.data_shape[1])

        if self.decoder_distribution == "ssim":
            _, value = ssim(x_hat, x, kernel_size=11, reduction=None, data_range=1.0, return_full_image=True)
            return -(1 - value).sum(dim=(1, 2, 3))

        if self.decoder_distribution == "bce":
            return -F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=(1, 2, 3))

        if self.decoder_distribution == "mse":
            return -F.mse_loss(x_hat, x, reduction="none").sum(dim=(0, 1, 2, 3))

        if self.decoder_distribution == "cross-entropy":


            # Because true images are not binary, the lower bound in the xent is not zero:
            # the lower bound in the xent is the entropy of the true images.

            loss_lower_bound = 0
            if self.subtract_true_image_entropy:
                x_clamp = torch.clamp(x, min=1e-6, max=1 - 1e-6)
                dist = torch.distributions.bernoulli.Bernoulli(probs=x_clamp)
                loss_lower_bound = torch.sum(dist.entropy(), dim=(1, 2, 3))

            eps = 1e-8
            x_hat_clamp = torch.clamp(x_hat, min=1e-6, max=1 - 1e-6)
            loss = -(x * torch.log(x_hat_clamp) + (1 - x) * torch.log(1 - x_hat_clamp)).sum(dim=(1, 2, 3))
            loss = (loss - loss_lower_bound)
            # loss = (x * torch.log(x_hat_clamp)).sum(dim=1)
            return -loss

        if self.decoder_distribution == "bernoulli":
            dist = torch.distributions.bernoulli.Bernoulli(probs=x_hat)

        elif self.decoder_distribution == "continuos_bernoulli":
            dist = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=x_hat)

        elif self.decoder_distribution == "gaussian":
            scale = torch.exp(logscale.to(self.device))
            mean = x_hat
            dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def save_state(self, path):
        ''' Save model state, including criterion and optimiizer '''
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'fc_mu_state_dict': self.fc_mu.state_dict(),
            'fc_var_state_dict': self.fc_var.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'loss': self._criterion,
        }, path)

    def load_state(self, path):
        ''' Load model state, including criterion and optimiizer '''

        # load the model checkpoint
        checkpoint = torch.load(path)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self.fc_mu.load_state_dict(checkpoint["fc_mu_state_dict"])
        self.fc_var.load_state_dict(checkpoint["fc_var_state_dict"])

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self._criterion = checkpoint['loss']