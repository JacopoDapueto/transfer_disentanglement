

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from src.methods.VAE.architecture import VAE




class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, x):
        batch_size = x.size(0)
        shape = (batch_size, *self.shape)
        out = x.view(shape)
        return out


class NormalizeTanh(nn.Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(input) + 1)/2




class EFFICIENTVAE(VAE):

    def __init__(self,  n_filters, decoder_distribution,  beta, data_shape, latent_dim, n_channel, **kwargs):
        super(EFFICIENTVAE, self).__init__(  n_filters, decoder_distribution,  beta, data_shape, latent_dim, n_channel, **kwargs)

        # encoded feature's size and volume
        self.feature_size = data_shape[0] // 8
        self.feature_volume = n_filters * (self.feature_size ** 2)

        # projection
        self.project = self._linear(latent_dim, self.feature_volume, relu=True)

        self.encoder = nn.Sequential(
            self._conv(n_channel, n_filters // 4),
            self._conv(n_filters // 4, n_filters // 2),
            self._conv(n_filters // 2, n_filters),
            View([self.feature_volume]),
            nn.LayerNorm(self.feature_volume)
        )



        self.decoder = nn.Sequential(
            self.project,
            View((n_filters, self.feature_size, self.feature_size)),
            self._deconv(n_filters, n_filters // 2),
            self._deconv(n_filters // 2, n_filters // 4),
            self._deconv(n_filters // 4, n_channel),
            NormalizeTanh()
        )

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        # distribution parameters
        self.fc_mu = self._linear(self.feature_volume, latent_dim, relu=False)
        self.fc_var = self._linear(self.feature_volume, latent_dim, relu=False)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.one = torch.Tensor([1.])

        self.clip_grad = 1.0 # clip value of gradient norm

        # linear warm-up
        self.beta = beta
        self.current_iteration = 0
        self.decoder_distribution = decoder_distribution # "gaussian" # "bernoulli"

        self.subtract_true_image_entropy=False

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.LeakyReLU(0.02),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.LeakyReLU(0.02),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(0.02),
        ) if relu else nn.Linear(in_size, out_size)




