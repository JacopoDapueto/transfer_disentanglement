
"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from src.data import utils
from src.data.factor_data_class import FactorData
from src.data.utils import set_transform

SHAPES3D_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", ""), "3dshapes", '3dshapes.h5')


class Shapes3D(FactorData):
  """Shapes3D dataset.

  The data set was originally introduced in "Disentangling by Factorising".

  The ground-truth factors of variation are (in the default setting):
  0 - floor_hue (10 different values)
  1 - wall_hue (10 different values)
  2 - object_hue (10 different values)
  3 - scale (8 different values)
  4 - shape (4 different values)
  5 - orientation (15 different values)
  """

  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None):

      super(Shapes3D, self).__init__(latent_factor_indices, batch_size, random_state)

      if latent_factor_indices is None:
          latent_factor_indices = list(range(6))

      self.latent_factor_indices = latent_factor_indices

      # Load the data so that we can sample from it.
      try:
          # Data was saved originally using python2, so we need to set the encoding.
          data = h5py.File(SHAPES3D_PATH, 'r')
      except:
          raise ValueError("Shapes3D dataset not found.")

      self.images = np.array(data['images']).astype(np.float32)
      self.factor_sizes = np.array([10, 10, 10, 8, 4, 15], dtype=np.int64)[self.latent_factor_indices]
      self.factor_names = [['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'][i] for i in self.latent_factor_indices]  # name of the factors
      self.latents_classes = np.array(data["labels"])

      # transform into labels
      for i in range(self.latents_classes.shape[1]):
          le = LabelEncoder()
          self.latents_classes[:, i] = le.fit_transform(self.latents_classes[:, i])




      # remove non-unique elements wrt factors
      _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
      self.images = self.images[idx_keep]

      self.latents_classes = self.latents_classes[idx_keep][:, self.latent_factor_indices]

      self.full_factor_sizes = [10, 10, 10, 8, 4, 15]
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
      return 3

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
      images, classes = self.sample_observations_from_factors_aux(factors)

      images = np.squeeze(images)
      images = self.transform(images.astype(np.uint8))
      classes = torch.from_numpy(classes)
      return images, classes

  def sample_observations_from_factors_aux(self, factors):
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