from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n+1):
        yield iterable[ndx:min(ndx + n, l)]



def freeze_all_params(module, freeze=True):
    freeze = not freeze
    for param in module.parameters():
        param.requires_grad = freeze

    return module


def unfreeze_child_params(module, children):


    modules = list(module.children())

    selected =[modules[i] for i in children ]
    for child in selected:
        #print(child.name, "is unfreezed")

        for param in child.parameters():
            param.requires_grad = True

    return module



def normalize_image(image, min_value, max_value):
    # Normalize the image
    normalized_image = (image - min_value) / (max_value - min_value)
    return normalized_image

def denormalize_image(normalized_image, min_value, max_value):
    # Denormalize the image
    denormalized_image = normalized_image * (max_value - min_value) + min_value
    return denormalized_image


def standardize_image(image,  std, mean):
    # Normalize the image
    normalized_image = torch.sub(image, mean ) / (std + 1e-7)
    return normalized_image


def destandardize_image(normalized_image, std, mean):
    # Denormalize the image
    denormalized_image = normalized_image * (std + 1e-7 ) + mean
    return denormalized_image