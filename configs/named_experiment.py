"""Provides named experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.config.AE.experiment import AE
from src.config.DCGAN.experiment import DCGAN
from src.config.NF.experiment import NF
from src.config.VAE.experiment import VAE
from src.config.coil100_augmented.experiment import COIL100AUGMENTED
from src.config.coil100_augmented_binary.experiment import COIL100AUGMENTEDBINARY
from src.config.coil100_augmented_binary_weak.experiment import WEAKCOIL100AUGMENTEDBINARY
from src.config.coil100_augmented_weak.experiment import WEAKCOIL100AUGMENTED
from src.config.coil100_augmented_weak_efficient.experiment import EFFICIENTWEAKCOIL100AUGMENTED
from src.config.coil100_augmented_weak_shallow.experiment import SHALLOWWEAKCOIL100AUGMENTED
from src.config.colors_dsprites.experiment import COLORDSPRITES
from src.config.colors_dsprites_unsupervised.experiment import COLORDSPRITESUNSUPERVISED
from src.config.cropped_icub_128.experiment import CROPPEDICUB128
from src.config.croppped_icub.experiment import CROPPEDICUB
from src.config.croppped_icub_20.experiment import CROPPEDICUB20
from src.config.dsprites.experiment import DSPRITES
from src.config.dsprites_gaussian.experiment import DSPRITESGAUSSIAN
from src.config.dsprites_weak.experiment import WEAKDSPRITES
from src.config.icub.experiment import ICUB
from src.config.imagenet_to_dsprites.experiment import ImagenetToDsprites
from src.config.noisy_dsprites.experiment import NOISYDSPRITES
from src.config.noisy_dsprites_color.experiment import NOISYCOLORDSPRITES
from src.config.pug_animal_weak.experiment import WEAKPUGANIMAL
from src.config.rgbd_objects.experiment import RGBDOBJECTS
from src.config.rgbd_objects_depth.experiment import RGBDOBJECTSDEPTH
from src.config.shapes3d_weak.experiment import WEAKSHAPE3D
from src.config.texture_dsprites.experiment import TEXTUREDSPRITES
from src.config.white_colors_dsprites_unsupervised.experiment import WHITECOLORDSPRITESUNSUPERVISED
from src.config.white_colors_dsprites_weak.experiment import WHITECOLORDSPRITES


def get_named_experiment(name):

    if name == "AE":
        return AE()

    if name == "NF":
        return NF()

    if name == "VAE":
        return VAE()

    if name == "DCGAN":
        return DCGAN()

    if name == "ICUB":
        return ICUB()

    if name == "DSPRITES":
        return DSPRITES()

    if name == "NOISYDSPRITES":
        return NOISYDSPRITES()

    if name == "NOISYCOLORDSPRITES":
        return NOISYCOLORDSPRITES()

    if name == "TEXTUREDSPRITES":
        return TEXTUREDSPRITES()

    if name == "WEAKDSPRITES":
        return WEAKDSPRITES()

    if name == "WHITECOLORDSPRITES":
        return WHITECOLORDSPRITES()

    if name == "WHITECOLORDSPRITESUNSUPERVISED":
        return WHITECOLORDSPRITESUNSUPERVISED()

    if name == "COLORDSPRITESUNSUPERVISED":
        return COLORDSPRITESUNSUPERVISED()

    if name == "COLORDSPRITES":
        return COLORDSPRITES()

    if name == "DSPRITESGAUSSIAN":
        return DSPRITESGAUSSIAN()

    if name == "CROPPEDICUB":
        return CROPPEDICUB()

    if name == "CROPPEDICUB128":
        return CROPPEDICUB128()

    if name == "CROPPEDICUB20":
        return CROPPEDICUB20()

    if name == "COIL100AUGMENTED":
        return COIL100AUGMENTED()

    if name == "WEAKCOIL100AUGMENTED":
        return WEAKCOIL100AUGMENTED()

    if name == "WEAKCOIL100AUGMENTEDBINARY":
        return WEAKCOIL100AUGMENTEDBINARY()

    if name == "SHALLOWWEAKCOIL100AUGMENTED":
        return SHALLOWWEAKCOIL100AUGMENTED()

    if name == "EFFICIENTWEAKCOIL100AUGMENTED":
        return EFFICIENTWEAKCOIL100AUGMENTED()

    if name == "COIL100AUGMENTEDBINARY":
        return COIL100AUGMENTEDBINARY()

    if name == "WEAKPUGANIMAL":
        return WEAKPUGANIMAL()

    if name == "RGBDOBJECTS":
        return RGBDOBJECTS()

    if name == "RGBDOBJECTSDEPTH":
        return RGBDOBJECTSDEPTH()

    if name == "WEAKSHAPE3D":
        return WEAKSHAPE3D()

    if name == "ImagenetToDsprites":
        return ImagenetToDsprites()


    raise ValueError("Invalid experiment name.")
