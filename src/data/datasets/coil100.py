"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import pandas as pd

import torch

import copy


from src.data.factor_data_class import FactorData
from src.data import utils



COIL100AUGMENTED_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", ""), "coil-100",
    "coil-100-augmented")


COIL100AUGMENTEDBINARY_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", ""), "coil-100",
    "coil-100-augmented-binary")



class Coil100Augmented(FactorData):
    """COIL-100 dataset.

  Augmented version of COIL-100
  https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php.

  The ground-truth factors of variation are (in the default setting):
  0 - object (100 different values)
  1 - pose (72 different values)
  2 - rotation (18 different values)
  3 - scale (9 different values)
  """

    def __init__(self, latent_factor_indices=None, batch_size=1, random_state=0,  resize=None, center_crop=None, **kwargs):
        # By default, all factors are considered ground truth
        # factors.

        super(Coil100Augmented, self).__init__(latent_factor_indices, batch_size, random_state)

        if latent_factor_indices is None:
            latent_factor_indices = list(range(4))

        self.latent_factor_indices = latent_factor_indices

        self.data_shape = [128, 128, 3]
        # Load the data so that we can sample from it.
        try:
            data = os.walk(COIL100AUGMENTED_PATH, topdown=False)
        except:
            raise ValueError("COIL-100 dataset not found.")

        classes_file = pd.read_csv(os.path.join(COIL100AUGMENTED_PATH, "classes.csv"))
        self.images = classes_file["image"]
        self.factor_sizes = np.array([100, 72, 18, 9], dtype=np.int64)
        self.factor_names = [list(classes_file.columns[1:])[i] for i in self.latent_factor_indices]  # name of the factors
        self.latents_classes = np.array(classes_file.loc[:, classes_file.columns != 'image'].values)

        del classes_file

        # remove non-unique elements wrt factors
        _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
        self.images = np.array([self.images[id] for id in idx_keep])
        self.latents_classes = np.array(self.latents_classes[idx_keep][:, self.latent_factor_indices])

        self.full_factor_sizes = [100, 72, 18, 9]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = copy.deepcopy(utils.FactorSampler(self.factor_sizes, self.latent_factor_indices))

        self.resize = resize
        self.center_crop = center_crop

        self.transform, self.data_shape = utils.set_transform(128, 3, resize, center_crop)


        print("Batch size of the Dataset is ignored...Dataloader will batch!")
        self.batch_size = 1

    def load_rgb_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.transform(img)

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
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

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

        classes = torch.from_numpy(classes)
        return torch.squeeze(images), classes

    def sample_observations_from_factors_aux(self, factors):
        """Sample a batch of observations X given a batch of factors Y.
       Return also the factor classes
        """

        all_factors = self.state_space.sample_all_factors(factors, self.random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)

        batch_imgs = self.load_rgb_img(os.path.join(COIL100AUGMENTED_PATH, self.images[indices][0]))
        return batch_imgs, self.latents_classes[indices]

    def get_images(self, index):

        batch_imgs = self.load_rgb_img(os.path.join(COIL100AUGMENTED_PATH, self.images[index]))

        classes = np.array(self.latents_classes[index])

        classes = classes[np.newaxis, ...]

        return torch.from_numpy(classes), batch_imgs, torch.from_numpy(classes)

    def _sample_factor(self, i):
        return self.random_state.randint(self.factor_sizes[i], size=self.batch_size)




class Coil100AugmentedBinary(Coil100Augmented):
    """COIL-100 dataset.

  Augmented version of COIL-100
  https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php.

  The ground-truth factors of variation are (in the default setting):
  0 - object (100 different values)
  1 - pose (72 different values)
  2 - rotation (18 different values)
  3 - scale (9 different values)
  """

    def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, **kwargs):
        # By default, all factors are considered ground truth

        super(Coil100AugmentedBinary, self).__init__(latent_factor_indices, batch_size, random_state)

        if latent_factor_indices is None:
            latent_factor_indices = list(range(4))

        self.latent_factor_indices = latent_factor_indices

        self.data_shape = [128, 128, 3]

        # Load the data so that we can sample from it.
        try:

            data = os.walk(COIL100AUGMENTEDBINARY_PATH, topdown=False)
        except:
            raise ValueError("COIL-100 dataset not found.")

        classes_file = pd.read_csv(os.path.join(COIL100AUGMENTEDBINARY_PATH, "classes.csv"))
        self.images = classes_file["image"]
        self.factor_sizes = np.array([100, 72, 18, 9], dtype=np.int64)
        self.factor_names = [list(classes_file.columns[1:])[i] for i in
                             self.latent_factor_indices]  # name of the factors
        self.latents_classes = np.array(classes_file.loc[:, classes_file.columns != 'image'].values)

        del classes_file

        # remove non-unique elements wrt factors
        _, idx_keep = np.unique(self.latents_classes[:, self.latent_factor_indices], axis=0, return_index=True)
        self.images = np.array([self.images[id] for id in idx_keep])
        self.latents_classes = np.array(self.latents_classes[idx_keep][:, self.latent_factor_indices])

        self.full_factor_sizes = [100, 72, 18, 9]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = copy.deepcopy(utils.FactorSampler(self.factor_sizes, self.latent_factor_indices))

        self.resize = resize
        self.center_crop = center_crop

        self.transform, self.data_shape = utils.set_transform(128, 3, resize, center_crop)

        print("Batch size of the Dataset is ignored...Dataloader will batch!")
        self.batch_size = 1


    def num_channels(self):
        return 3

    def sample_observations_from_factors_aux(self, factors):
        """Sample a batch of observations X given a batch of factors Y.
       Return also the factor classes
        """

        all_factors = self.state_space.sample_all_factors(factors, self.random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)

        batch_imgs = self.load_rgb_img(os.path.join(COIL100AUGMENTEDBINARY_PATH, self.images[indices][0]))
        return batch_imgs, self.latents_classes[indices]

    def get_images(self, index):

        batch_imgs = self.load_rgb_img(os.path.join(COIL100AUGMENTEDBINARY_PATH, self.images[index]))

        classes = np.array(self.latents_classes[index])

        classes = classes[np.newaxis, ...]

        return torch.from_numpy(classes), batch_imgs, torch.from_numpy(classes)

