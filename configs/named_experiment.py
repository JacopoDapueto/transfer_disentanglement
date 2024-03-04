"""Provides named experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from configs.coil100_augmented.experiment import COIL100AUGMENTED
from configs.coil100_augmented_binary.experiment import COIL100AUGMENTEDBINARY
from configs.coil100_augmented_binary_weak.experiment import WEAKCOIL100AUGMENTEDBINARY
from configs.coil100_augmented_weak.experiment import WEAKCOIL100AUGMENTED
from configs.colors_dsprites_weak.experiment import WEAKCOLORDSPRITES
from configs.colors_dsprites.experiment import COLORDSPRITES
from configs.noisy_dsprites.experiment import NOISYDSPRITES
from configs.noisy_dsprites_color.experiment import NOISYCOLORDSPRITES
from configs.rgbd_objects.experiment import RGBDOBJECTS
from configs.rgbd_objects_depth.experiment import RGBDOBJECTSDEPTH
from configs.shapes3d_weak.experiment import WEAKSHAPE3D
from configs.shapes3d.experiment import SHAPE3D

from configs.white_colors_dsprites_weak.experiment import WHITECOLORDSPRITES


def get_named_experiment(name):

    if name == "NOISYDSPRITES":
        return NOISYDSPRITES()

    if name == "NOISYCOLORDSPRITES":
        return NOISYCOLORDSPRITES()

    if name == "WHITECOLORDSPRITES":
        return WHITECOLORDSPRITES()

    if name == "COLORDSPRITES":
        return COLORDSPRITES()

    if name == "WEAKCOLORDSPRITES":
        return WEAKCOLORDSPRITES()

    if name == "COIL100AUGMENTED":
        return COIL100AUGMENTED()

    if name == "WEAKCOIL100AUGMENTED":
        return WEAKCOIL100AUGMENTED()

    if name == "WEAKCOIL100AUGMENTEDBINARY":
        return WEAKCOIL100AUGMENTEDBINARY()

    if name == "COIL100AUGMENTEDBINARY":
        return COIL100AUGMENTEDBINARY()

    if name == "RGBDOBJECTS":
        return RGBDOBJECTS()

    if name == "RGBDOBJECTSDEPTH":
        return RGBDOBJECTSDEPTH()

    if name == "WEAKSHAPE3D":
        return WEAKSHAPE3D()

    if name == "SHAPE3D":
        return SHAPE3D()



    raise ValueError("Invalid experiment name.")
