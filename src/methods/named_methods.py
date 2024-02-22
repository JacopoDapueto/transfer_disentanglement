"""Provides named methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.methods.AE.architecture import AE
from src.methods.EFFICIENT_VAE.architecture import EFFICIENTVAE
from src.methods.EFFICIENT_VAE_weak.architecture import EFFICIENTWEAKVAE
from src.methods.VAE.architecture import VAE
from src.methods.VAE_weak.architecture import WEAKVAE


def get_named_method(name):

    if name == "AE":
        return AE
    elif name == "VAE":
        return VAE

    elif name == "WEAKVAE":
        return WEAKVAE

    elif name == "EFFICIENTWEAKVAE":
        return EFFICIENTWEAKVAE
    elif name == "EFFICIENTVAE":
        return EFFICIENTVAE

    else:
        raise ValueError("Invalid method name.")

