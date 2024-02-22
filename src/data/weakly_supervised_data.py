

"""Main training protocol used for weakly-supervised disentanglement models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch

from src.data.factor_data_class import FactorData, ShuffledSubset
from src.data.named_data import get_named_data


class WeaklySupervisedData(FactorData):

  def __init__(self, dataset_name ,factors_idx, batch_size=1,  random_state=0, k=1, resize=None, center_crop=None):

    if isinstance(random_state, int):
      random_state = np.random.default_rng(seed=random_state)


    self.dl = get_named_data(dataset_name)(factors_idx, batch_size, random_state, resize=resize, center_crop=center_crop)
    self.k = k


  def num_channels(self):
    return self.dl.num_channels()

  def get_shape(self):
    return self.dl.get_shape()

  @property
  def num_images(self):
    return self.dl.num_images

  @property
  def batch_size(self):
    return self.dl.batch_size

  @property
  def num_factors(self):
    """ Number of factors of variantions    """

    return self.dl.num_factors

  @property
  def factors_sizes(self):
    """ Number of values for each factors (only considered)"""
    return self.dl.factors_sizes

  @property
  def full_factors_sizes(self):
    """ Number of values for each factors"""
    return self.dl.full_factors_sizes

  @property
  def observation_shape(self):
    return self.dl.observation_shape

  def sample_factors(self):
    """Sample a batch of factors Y."""
    return self.dl.sample_factors()

  def sample_observations_from_factors(self, factors):
    """Sample a batch of observations X given a batch of factors Y."""
    return self.dl.sample_observations_from_factors(factors)

  def sample(self):
    """Sample a batch of factors Y and observations X.
    Then sample X_next with k different factors """
    factors = self.sample_factors()
    images, classes = self.sample_observations_from_factors(factors)

    return torch.from_numpy(factors), images, classes

  def sample_observations(self):
    """Sample a batch of observations X."""
    factors, images, classes = self.sample()
    return images

  def get_images(self, index):
    """Get image and classes given index."""
    return self.dl.get_images(index)

  def __getitem__(self, index):
    '''Load image.
            	   Args:
              		idx: (int) image index.
            	   Returns:
              		img: (tensor) image tensor.
              		labels: (tensor) class label targets.
    '''
    # index is ignored
    factors, images, classes =  self.get_images(index) # self.sample()


    # sample next factor
    next_factors, index = self.get_next_factor(factors, return_index=True)
    next_images, next_classes = self.sample_observations_from_factors(next_factors)

    label = index


    return images, next_images, classes, next_classes, torch.tensor(label,  dtype=torch.int64)


  def __len__(self):
    return len(self.dl)

  def __iter__(self):

    self.start = 0  # the self.images are already filtered and index starts from zero
    self.end = self.dl.images.shape[0]

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None:  # single-process data loading, return the full iterator

      iter_start = self.start
      iter_end = self.end
    else:  # in a worker process
      # split workload

      per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))

      worker_id = worker_info.id

      iter_start = self.start + worker_id * per_worker

      iter_end = min(iter_start + per_worker, self.end)

    indicies = list(range(iter_start, iter_end))
    subset = ShuffledSubset(self, indicies)

    return iter(subset)


  def get_next_factor(self, f, return_index=False):
    """Given FoV f, return the sample with k different FoV."""

    z = f.numpy().copy()

    if self.k == -1:
      k_observed = self.dl.random_state.randint(1, self.num_factors)
    else:
      k_observed = self.k

    index_list = self.dl.random_state.choice(z.shape[1], self.dl.random_state.choice([1, k_observed]), replace=False)
    idx = -1

    for index in index_list:

      # sample next factor
      new = np.random.choice(range(self.factors_sizes[index]))

      z[:, index] = new
      idx = index

    if return_index:
      return z, idx

    return z, k_observed


