

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torchsummary import summary

from src.methods.VAE.architecture import VAE

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

    def __init__(self,  n_filters, decoder_distribution,  beta, data_shape, latent_dim, n_channel, criterion, **kwargs):
        super(EFFICIENTVAE, self).__init__(  n_filters, decoder_distribution,  beta, data_shape, latent_dim, n_channel, criterion, **kwargs)

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

        # for the gaussian likelihoo+d
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
            #nn.BatchNorm2d(kernel_num),
            nn.LeakyReLU(0.02),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(0.02),
        ) if relu else nn.Linear(in_size, out_size)






if __name__ == "__main__":
    args = {'dataset': 'coil100_augmented', 'decoder_distribution': 'gaussian', 'batch_size': 64, 'lr': 0.0001,
            'wd': 0.0,
            'epochs': 17, 'loss': 'mse', 'factor_idx': [0, 1, 2, 3], 'method': 'EFFICIENTWEAKVAE', 'beta': 2.0, 'n_filters': 64,
            'latent_dim': 30, 'random_seed': 3, 'aggregator': 'labels', 'k': 1, "data_shape": [128, 128],
            "n_channel": 3, "warm_up_iterations": 0}

    model = EFFICIENTVAE(**args, criterion=nn.MSELoss(reduction='sum')).to(device)
    print("Number of parameters of the model: {:,}".format(model.num_params()))

    input = torch.randn([64, 1, 128, 128])
    summary(model, (3, 128, 128))


    #out = model.encoder(input)

    #print(out.size())

    #input = torch.randn([64, args["latent_dim"]])

    #out = model.decoder(input)