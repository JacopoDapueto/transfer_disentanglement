
"""DSprites dataset ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
import torch
import torchvision
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from PIL import Image

from src.data.factor_data_class import FactorData
from src.data import utils

from src.data.utils import set_transform

DSPRITES_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

TEXTURE_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "texture")


BRODATZTEXTURE_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "texture", "Normalized_Brodatz")


REPRESENTATIONDSPRITES_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
    "representation", "vit-b-1k-dino")




def factor_to_index(factors, factors_size):
    # Inverts the behavior of the original function.

    idx = 0
    for factor, size in zip(factors, factors_size):
        idx = idx * size + factor
    return idx

def index_to_factor(idx, factors_size):

    factors = []
    for i, size in enumerate(factors_size):

        factor_idx = idx % size
        idx = idx // size

        factors.insert(0, factor_idx)  # Insert at the beginning of the list

    return factors


def classes_to_index(sizes, labels):
    """
    Given a list of class sizes and a corresponding list of labels, return the index of the image.

    :param sizes: List of sizes representing the number of classes for each attribute.
    :param labels: List of class labels (one for each attribute).
    :return: Index of the image corresponding to the class labels.
    """
    assert len(sizes) == len(labels), "Sizes and labels must have the same length."

    index = 0
    # Multiply class labels with the product of all subsequent sizes
    product = 1
    for i in reversed(range(len(sizes))):
        index += labels[i] * product
        product *= sizes[i]

    return index


def index_to_classes(sizes, index):
    """
    Given an index, return the corresponding class labels.

    :param sizes: List of sizes representing the number of classes for each attribute.
    :param index: Index of the image.
    :return: List of class labels corresponding to the given index.
    """
    labels = []

    for size in reversed(sizes):
        labels.append(index % size)
        index //= size

    return labels[::-1]  # Reverse the list to get the correct order



def load_brodatz_texture(factor):

  factor_to_image = {0: "D11", 1: "D40", 2:"D23", 3:"D109", 4:"D102"}
  texture = cv2.imread(os.path.join(BRODATZTEXTURE_PATH, factor_to_image[factor] + ".tif"), cv2.IMREAD_GRAYSCALE) / 255.


  bias = 20
  cropped = texture[bias:224 + bias, bias:224 + bias]

  return cropped



# Object colors generated using
# >> seaborn.husl_palette(n_colors=6, h=0.1, s=0.7, l=0.7)
OBJECT_COLORS = [[1., 1., 1.],
     [0.9096231780824386, 0.5883403686424795, 0.3657680693481871],
     [0.6350181801577739, 0.6927729880940552, 0.3626904230371999],
     [0.3764832455369271, 0.7283900430001952, 0.5963114605342514],
     [0.39548987063404156, 0.7073922557810771, 0.7874577552076919],
     [0.6963644829189117, 0.6220697032672371, 0.899716387820763],
     [0.90815966835861, 0.5511103319168646, 0.7494337214212151]]

BACKGROUND_COLORS = np.array([
    (0., 0., 0.),
    (.25, .25, .25),
    (.5, .5, .5),
    (.75, .75, .75),
    (1., 1., 1.),
])


def apply_color(image, color_idx):
    image = np.expand_dims(image, axis=0)
    observations = np.repeat(image, 3, axis=0)

    color = np.tile(np.array(OBJECT_COLORS)[color_idx].reshape(3, 1, 1),
                    [1, observations.shape[1], observations.shape[2]])

    new_images = observations * color


    return new_images


def apply_texture(image, texture):

    return image * texture



class TextureDSprites(data.Dataset): # data.Dataset
  """DSprites dataset.

  The data set was originally introduced in "beta-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework" and can be downloaded from
  https://github.com/deepmind/dsprites-dataset.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):
    # By default, all factors (including shape) are considered ground truth
    # factors.

    super(TextureDSprites, self).__init__()

    if latent_factor_indices is None:
      latent_factor_indices = list(range(7))

    self.latent_factor_indices = latent_factor_indices

    # Load the data so that we can sample from it.
    try:
        # Data was saved originally using python2, so we need to set the encoding.
        data = np.load(DSPRITES_PATH, encoding="latin1", allow_pickle=True)
    except:
        raise ValueError("DSprites dataset not found.")

    self.images = np.array(data["imgs"])
    self.latents_classes = np.array(data["latents_classes"])[:, 1:]

    self.texture_size = 5
    self.color_size = 7
    self.full_factor_sizes = np.array([5 ,7, 3, 6, 40, 32, 32])
    self.factor_names = ["Texture", "Color", "Shape", "Scale", "Orientation", "PosX", "PosY"]

    self.resize = resize
    self.center_crop = center_crop

    self.resize = 224  # 64
    self.transform, self.data_shape = set_transform(64, 3, self.resize, center_crop)

    self.data_shape = [224, 224, 3]

    self.textures_list = [load_brodatz_texture(i) for i in range(self.texture_size)]


  def num_factors(self):
      return 7


  def num_channels(self):
      return 3


  def get_shape(self):
      return self.data_shape


  def __len__(self):
    return np.prod(self.full_factor_sizes)


  def __getitem__(self, idx):

      factors= index_to_classes(self.full_factor_sizes, idx)

      # texture to apply
      texture_idx = factors[0]

      # color to apply
      color_idx = factors[ 1]

      idx_= classes_to_index( self.full_factor_sizes[2:], factors[ 2:] )
      image, classes = self.images[idx_], self.latents_classes[idx_]

      # add color
      image = apply_color(image, color_idx)

      # make 224x224
      image = np.moveaxis((255 * image).astype(np.uint8), 0, -1)

      image = torch.squeeze(self.transform(image))

      # add texture
      image = apply_texture(image, self.textures_list[texture_idx])

      # add texture and color classes
      classes  = np.concatenate([[texture_idx], [color_idx], classes], axis=0)

      factors = np.expand_dims(classes, axis=0)
      factors = torch.from_numpy(factors)

      return factors, image.float(), classes



  def sample_observations_from_factors(self, factors):

      factors = factors[0]
      idx = classes_to_index( self.full_factor_sizes, factors)
      _ , image, classes = self.__getitem__(idx)
      return image, classes


  def get_images(self, idx):
      return self.__getitem__(idx)


  @property
  def factors_sizes(self):
      return self.full_factor_sizes


