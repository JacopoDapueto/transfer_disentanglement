"""
Abstract class for data sets that are two-step generative models.
First sample factor of variation and the sample image given factors
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torch.utils.data as data


class ShuffledSubset(data.Subset):
    def __init__(self, dataset, indices, shuffle=True):
        super(ShuffledSubset, self).__init__(dataset, indices)
        self.shuffle = shuffle
        self.dataset = dataset

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        for index in indices:
            yield self.dataset[index]



class FactorData(data.IterableDataset):
    """Abstract class for data sets that are two-step generative models."""

    def __init__(self, factors_idx, batch_size=64, random_state=0, resize=None):
        super(FactorData, self).__init__()

        if isinstance(random_state, int):
            self.random_state = np.random.default_rng(seed=random_state)
        else:
            self.random_state = random_state

        self.factors_idx = factors_idx  # indices factor to consider
        self.batch_size = batch_size

        self.images = None
        self.latents_classes = None

    def num_channels(self):
        raise NotImplementedError()

    def get_shape(self):
        raise NotImplementedError()

    @property
    def num_images(self):
        raise NotImplementedError()

    @property
    def num_factors(self):
        """ Number of factors of variantions    """

        raise NotImplementedError()

    @property
    def factors_sizes(self):
        """ Number of values for each factors (only considered)"""
        raise NotImplementedError()

    @property
    def full_factors_sizes(self):
        """ Number of values for each factors"""
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors()
        images, classes = self.sample_observations_from_factors(factors)
        return factors, images, classes

    def sample_observations(self):
        """Sample a batch of observations X."""
        factors, images, classes = self.sample()
        return images

    def get_images(self, index):
        """Get image and classes given index."""
        raise NotImplementedError()

    def __getitem__(self, index):
        '''Load image.
            	   Args:
              		idx: (int) image index.
            	   Returns:
              		img: (tensor) image tensor.
              		labels: (tensor) class label targets.
    '''

        # index is ignored
        factors, images, classes = self.get_images(index)

        return images, classes

    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __iter__(self):

        self.start = 0 # the self.images are already filtered and index starts from zero
        self.end = self.images.shape[0]

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


