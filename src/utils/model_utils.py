"""
https://stackoverflow.com/questions/50879438/sparsity-reduction
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def _conv( channel_size, kernel_num, kernel_size=4, stride=2, padding=1, bn=True, relu=True):
    layers = [nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )]
    if bn:
        layers.append(nn.BatchNorm2d(kernel_num))
    if relu:
        layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)


def _deconv( channel_num, kernel_num, kernel_size=4, stride=2, padding=1, output_padding=0,  bn=True, relu=True):

    layers = [nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )]

    if bn:
        layers.append(nn.BatchNorm2d(kernel_num))
    if relu:
        layers.append(nn.LeakyReLU(0.2))

    return nn.Sequential(*layers)


def _linear( in_size, out_size, relu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LeakyReLU(0.2),
    ) if relu else nn.Linear(in_size, out_size)


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




# Define a custom module to resize the input image
class ResizeLayer(nn.Module):
    def __init__(self, size=(224, 224)):
        super(ResizeLayer, self).__init__()
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=False)




