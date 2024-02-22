"""Utils functions for data set code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder


class FactorSampler(object):
  """State space with factors split between latent variable and observations."""

  def __init__(self, factor_sizes, latent_factor_indices):
    self.factor_sizes = factor_sizes
    self.num_factors = len(self.factor_sizes)
    self.latent_factor_indices = range(self.num_factors) #latent_factor_indices
    self.observation_factor_indices = [
        i for i in range(self.num_factors)
        if i not in self.latent_factor_indices
    ]

  @property
  def num_latent_factors(self):
    return self.num_factors

  def sample_latent_factors(self, num, random_state):
    """Sample a batch of the latent factors. shape = (num observations, num factors)"""

    factors = np.hstack([np.expand_dims(self._sample_factor(i, num, random_state), axis=-1) for i in self.latent_factor_indices])

    return factors

  def sample_all_factors(self, latent_factors, random_state):
    """Samples the remaining factors based on the latent factors."""

    num_samples = latent_factors.shape[0]
    all_factors = np.zeros(
        shape=(num_samples, self.num_factors), dtype=np.int64)
    all_factors[:, self.latent_factor_indices] = latent_factors

    # Complete all the other factors
    for i in self.observation_factor_indices:
      all_factors[:, i] = self._sample_factor(i, num_samples, random_state)

    return all_factors

  def _sample_factor(self, i, num, random_state):
    return random_state.integers(self.factor_sizes[i], size=num)




# Custom transformer class
class WordToCategoryTransformer:
  def __init__(self):
    # Initialize a LabelEncoder
    self.label_encoder = LabelEncoder()
    self.fitted = False

  def fit(self, arr):
    # Fit the label encoder on the input array
    self.label_encoder.fit(arr)
    self.fitted = True

  def transform(self, arr):
    # Transform the input array based on the fitted label encoder
    if not self.fitted:
      raise ValueError("Transformer must be fitted before transformation.")
    return self.label_encoder.transform(arr)


  def inverse_transform(self, arr):
    # Transform the input array based on the fitted label encoder
    if not self.fitted:
      raise ValueError("Transformer must be fitted before transformation.")


    return self.label_encoder.inverse_transform(arr)

  def fit_transform(self, arr):
    # Fit and transform in a single step
    self.fit(arr)
    return self.transform(arr)



def index_to_factor(index, factor_sizes, batch_size):
  factors = np.zeros((batch_size, factor_sizes.shape[0]), dtype=np.int64)

  new_index = np.asarray(index)
  for i, size in enumerate(factor_sizes):
    factors[:, (factor_sizes.size - 1) - i] = np.remainder(new_index, size)

    new_index = np.subtract(new_index, np.divide(new_index, size))


  return factors.astype(np.int64)




def set_transform(default_shape, channels, resize=None, center_crop=None):


  transform = [T.ToPILImage(), T.ToTensor()]


  if center_crop is not None:

      default_shape = center_crop
      transform.append(T.CenterCrop(center_crop))

  if resize is not None:

    default_shape = resize
    transform.append(T.Resize([resize, resize], antialias=True))

  data_shape = [default_shape, default_shape, channels]

  transform = T.Compose(transform)

  return transform, data_shape
