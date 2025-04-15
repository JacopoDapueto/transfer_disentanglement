"""Provides named data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.data.datasets.coil100 import Coil100Augmented
from src.data.datasets.coil100 import Coil100AugmentedBinary
from src.data.datasets.dsprites import ColorDSprites
from src.data.datasets.dsprites import DSprites, NoisyDSprites, NoisyColorDSprites
from src.data.datasets.dsprites import WhiteColorDSprites
from src.data.datasets.rgbd_objects import RGBDObjects
from src.data.datasets.rgbd_objects import RGBDObjectsDepth
from src.data.datasets.shapes3d import Shapes3D
from src.data.datasets.texture_dsprites import TextureDSprites


def get_named_data(name):

    if name == "dsprites":
        return DSprites

    if name == "noisy-dsprites":
        return NoisyDSprites

    if name == "texture-dsprites":
        return TextureDSprites

    if name == "3dshapes":
        return Shapes3D

    if name == "dsprites-color":
        return ColorDSprites

    if name == "noisy-dsprites-color":
        return NoisyColorDSprites

    if name == "dsprites-color-white":
        return WhiteColorDSprites

    if name == "coil100_augmented":
        return Coil100Augmented

    if name == "rgbd_objects":
        return RGBDObjects

    if name == "rgbd_objects_depth":
        return RGBDObjectsDepth

    if name == "coil100_augmented_binary":
        return Coil100AugmentedBinary


    raise ValueError("Invalid data name.")
