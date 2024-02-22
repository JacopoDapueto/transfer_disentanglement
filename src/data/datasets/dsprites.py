
"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch

from src.data import utils
from src.data.factor_data_class import FactorData
from src.data.utils import set_transform

DSPRITES_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", ""), "dsprites",
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

TEXTURE_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", ""), "texture")


class DSprites(FactorData):
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

    super(DSprites, self).__init__(latent_factor_indices, batch_size, random_state)

    if latent_factor_indices is None:
      latent_factor_indices = list(range(6))

    self.latent_factor_indices = latent_factor_indices


    # Load the data so that we can sample from it.
    try:
      # Data was saved originally using python2, so we need to set the encoding.
      data = np.load(DSPRITES_PATH, encoding="latin1", allow_pickle=True)
    except:
      raise ValueError("DSprites dataset not found.")

    self.images = np.array(data["imgs"])
    self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)[self.latent_factor_indices]
    self.factor_names = [list(data["metadata"][()]["latents_names"])[i] for i in self.latent_factor_indices] # name of the factors
    self.latents_classes = np.array(data["latents_classes"])

    # remove non-unique elements wrt factors
    _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
    self.images = self.images[idx_keep]

    self.latents_classes = self.latents_classes[idx_keep][:, self.latent_factor_indices]

    self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)

    self.state_space = utils.FactorSampler(self.factor_sizes, self.latent_factor_indices)

    self.resize = resize
    self.center_crop = center_crop

    self.transform, self.data_shape = set_transform(64, 3, resize, center_crop)

    print("Batch size of the Dataset is ignored...Dataloader will batch!")
    self.batch_size = 1



  def num_images(self):
    return self.latents_classes.shape[0]

  def num_channels(self):
    return 1

  def get_shape(self):
    return self.data_shape

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_sizes(self):
    return self.factor_sizes

  @property
  def full_factors_sizes(self):
    return self.full_factor_sizes

  @property
  def observation_shape(self):
    return self.data_shape

  def sample_factors(self):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(self.batch_size, self.random_state)

  def sample_observations_from_factors(self, factors):
    images, classes = self.sample_observations_from_factors_no_color(factors)

    images = self.transform(np.moveaxis(images, 0, -1))
    classes = torch.from_numpy(classes)
    return images, classes

  def sample_observations_from_factors_no_color(self, factors):
    """Sample a batch of observations X given a batch of factors Y.
       Return also the factor classes
    """

    indices = np.array(np.dot(factors, self.factor_bases), dtype=np.int64)

    return self.images[indices].astype(np.float32), self.latents_classes[indices]

  def get_images(self, index):

    factors = utils.index_to_factor(index, self.factor_sizes[::-1], self.batch_size)
    images, classes = self.sample_observations_from_factors(factors)
    return torch.from_numpy(factors), images, classes

  def _sample_factor(self, i):
    return self.random_state.integers(self.factor_sizes[i], size=self.batch_size)




class NoisyDSprites(DSprites):
  """Noisy DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the background pixels are replaced with random
  noise.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):
    super(NoisyDSprites, self).__init__(latent_factor_indices, batch_size, random_state, resize=resize, center_crop=center_crop, **kwargs)

  def sample_observations_from_factors(self, factors):
    images, classes = self.sample_observations_from_factors_no_color(factors)
    observations = np.repeat(images, 3, axis=0)

    color = self.random_state.uniform(0, 1, [3, observations.shape[1], observations.shape[2]])

    images  = (255 * np.minimum(observations + color, 1.)).astype(np.uint8)
    images = self.transform(np.moveaxis(images, 0, -1))
    classes = torch.from_numpy(classes)

    return images, classes

  def num_channels(self):
    return 3


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


class ColorDSprites(DSprites):
  """Color DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the sprite is colored in a randomly sampled
  color.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):

    super(ColorDSprites, self).__init__(latent_factor_indices, batch_size, random_state, resize=resize, center_crop=center_crop,  **kwargs)


    if 0 not in latent_factor_indices:
      raise "Cannot eliminate Color FoV from dsprites-color"

    if latent_factor_indices is None:
      latent_factor_indices = list(range(6))

    self.latent_factor_indices = latent_factor_indices

    # Load the data so that we can sample from it.
    try:
      # Data was saved originally using python2, so we need to set the encoding.
      data = np.load(DSPRITES_PATH, encoding="latin1", allow_pickle=True)
    except:
      raise ValueError("DSprites dataset not found.")

    self.images = np.array(data["imgs"])
    self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)[self.latent_factor_indices]

    # add 7 colors
    self.color_size = 7

    self.factor_names = [list(data["metadata"][()]["latents_names"])[i] for i in
                         self.latent_factor_indices]  # name of the factors
    self.latents_classes = np.array(data["latents_classes"])

    # remove non-unique elements wrt factors
    _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
    self.images = self.images[idx_keep]

    self.latents_classes = self.latents_classes[idx_keep][:, self.latent_factor_indices]

    self.full_factor_sizes = [self.color_size, 3, 6, 40, 32, 32]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.state_space = utils.FactorSampler(self.factor_sizes, self.latent_factor_indices)

    if 0 in self.latent_factor_indices:
      self.factor_sizes[0] = self.color_size

    self.color_factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.color_state_space = utils.FactorSampler(self.factor_sizes, self.latent_factor_indices)

    self.resize = resize
    self.center_crop = center_crop

    self.transform, self.data_shape = set_transform(64, 3, resize, center_crop)

    print("Batch size of the Dataset is ignored...Dataloader will batch!")
    self.batch_size = 1



  def sample_observations_from_factors(self, factors):


    no_color_factors = np.concatenate([np.zeros((factors.shape[0], 1)), factors[:,1:]], axis=1)

    no_color_observations, classes = self.sample_observations_from_factors_no_color(no_color_factors)


    observations = np.repeat(no_color_observations, 3, axis=0)

    color = np.tile(np.array(OBJECT_COLORS)[factors[:,0]].reshape(observations.shape[0], 1, 1), [ 1, observations.shape[1], observations.shape[2]])

    new_images = observations * color

    images = torch.squeeze(self.transform(np.moveaxis((255 * new_images).astype(np.uint8), 0, -1)))
    return images, torch.from_numpy(np.concatenate([factors[:,0][...,np.newaxis], classes[:,1:]], axis=1))


  def sample_factors(self):
    """Sample a batch of factors Y."""
    # add color
    no_color_factors = self.state_space.sample_latent_factors(self.batch_size, self.random_state)
    colors = self._sample_factor(0)[:, np.newaxis]
    return np.concatenate([colors, no_color_factors[:, 1:]], axis=1)


  def get_images(self, index):

    factors = utils.index_to_factor(index, self.factor_sizes[::-1], self.batch_size)

    images, classes = self.sample_observations_from_factors(factors)
    return torch.from_numpy(factors), torch.squeeze(images), classes

  def num_channels(self):
    return 3

  def num_images(self):
    prod = self.color_size * super(ColorDSprites, self).num_images()
    return np.int32(prod)


class NoisyColorDSprites(ColorDSprites):
  """Noisy DSprites.

  This data set is the same as the original DSprites data set except that when
  sampling the observations X, the background pixels are replaced with random
  noise.

  The ground-truth factors of variation are (in the default setting):
  0 - shape (3 different values)
  1 - scale (6 different values)
  2 - orientation (40 different values)
  3 - position x (32 different values)
  4 - position y (32 different values)
  """

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):
    super(NoisyColorDSprites, self).__init__(latent_factor_indices, batch_size, random_state, resize=resize, center_crop=center_crop, **kwargs)

  def sample_observations_from_factors(self, factors):

    no_color_factors = np.concatenate([np.zeros((factors.shape[0], 1)), factors[:, 1:]], axis=1)

    no_color_observations, classes = self.sample_observations_from_factors_no_color(no_color_factors)

    observations = np.repeat(no_color_observations, 3, axis=0)

    color = np.tile(np.array(OBJECT_COLORS)[factors[:, 0]].reshape(observations.shape[0], 1, 1),
                    [1, observations.shape[1], observations.shape[2]])


    new_images = observations * color

    noise = self.random_state.uniform(0, 1, [3, new_images.shape[1], new_images.shape[2]])

    new_images = np.minimum(new_images + (noise *   (observations - 1.)), 1.)


    images = self.transform(np.moveaxis((255 * new_images).astype(np.uint8), 0, -1))

    classes = np.concatenate([factors[:, 0][..., np.newaxis], classes[:, 1:]], axis=1)
    classes = torch.from_numpy(classes)

    return torch.squeeze(images), classes



class WhiteColorDSprites(ColorDSprites):

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):

    super(WhiteColorDSprites, self).__init__(latent_factor_indices, batch_size, random_state, **kwargs)

    if 0 not in latent_factor_indices:
      raise "Cannot eliminate Color FoV from dsprites-color"

    if latent_factor_indices is None:
      latent_factor_indices = list(range(6))

    self.latent_factor_indices = latent_factor_indices

    # Load the data so that we can sample from it.
    try:
      # Data was saved originally using python2, so we need to set the encoding.
      data = np.load(DSPRITES_PATH, encoding="latin1", allow_pickle=True)
    except:
      raise ValueError("DSprites dataset not found.")

    self.images = np.array(data["imgs"])
    self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)[self.latent_factor_indices]

    # add 7 colors
    self.color_size = 1

    self.factor_names = [list(data["metadata"][()]["latents_names"])[i] for i in
                         self.latent_factor_indices]  # name of the factors
    self.latents_classes = np.array(data["latents_classes"])

    # remove non-unique elements wrt factors
    _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
    self.images = self.images[idx_keep]

    self.latents_classes = self.latents_classes[idx_keep][:, self.latent_factor_indices]

    self.full_factor_sizes = [self.color_size, 3, 6, 40, 32, 32]
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.state_space = utils.FactorSampler(self.factor_sizes, self.latent_factor_indices)
    self.factor_sizes[0] = self.color_size
    # print(self.factor_sizes.shape)

    self.color_factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
    self.color_state_space = utils.FactorSampler(self.factor_sizes, self.latent_factor_indices)

    self.resize = resize
    self.center_crop = center_crop

    self.transform, self.data_shape = set_transform(64, 3, resize, center_crop)

    print("Batch size of the Dataset is ignored...Dataloader will batch!")
    self.batch_size = 1


