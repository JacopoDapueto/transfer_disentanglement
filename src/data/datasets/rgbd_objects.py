from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import pandas as pd
import re
import torch
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image

import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import albumentations as A

from sklearn.preprocessing import LabelEncoder

from src.data.factor_data_class import FactorData
from src.data.datasets.dsprites import DSprites
from src.data import utils

RGBDOBJECTS_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_TRANSFER_DATA", ""), "rgbd-objects")


def SquarePad( image, **params):
    w, h, c = image.shape
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = [hp, vp, hp, vp]
    image_padded = F.pad(Image.fromarray(image), padding, 0, 'constant')
    return np.asarray(image_padded)



def loader_rgb(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def repeat_3channels(x):
    return np.tile(x, [1, 1, 3])


def loader_binary(file):
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    mask = np.expand_dims(mask, axis=-1)

    mask = repeat_3channels(mask)

    return mask


class HierarchicalImageFolder(Dataset):

    def __init__(self, root, hlevel=2, ftype='_crop.png', transforms=None):
        '''
        A version of torchvision.datasets.ImageFolder adapted to images in
        a hierarchical file structure (i.e iNat18). The __getitem__(x) method
        returns the x-th image, along with a tuple for the hierarchical file label
        '''

        self.root = root

        self.n_categories = 51
        # for each category a different label encoder (number of instaces changes from category to category)
        self.categories_label_encoder = None
        self.instances_label_encoders = []

        self.imgs, self.classes = self.read_nested_images(root, ftype, hlevel)


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img_path, (category, instance) = self.imgs[index], self.classes[index]

        category_idx = self.categories_label_encoder.transform([category])[0]
        instance_idx = self.instances_label_encoders[category_idx].transform([instance])[0]

        rr = re.findall(r'\d+', img_path)  # first is number of instance
        azimuth_idx = int(rr[-2]) - 1
        pose_idx = int(rr[-1]) - 1

        img = loader_rgb(img_path)

        # load mask
        mask_path = img_path[:-8] + "maskcrop.png"
        mask = loader_binary(mask_path)

        if self.albumentation_transform is not None:
            augmented = self.albumentation_transform(image=img, mask=mask)

            img = augmented['image']
            mask = augmented['mask']

        # mask object
        img = img * (mask // np.max(mask))

        labels = np.array([category_idx, instance_idx, azimuth_idx, pose_idx])
        return self.transform(img), torch.from_numpy(labels)

    # transform classes to labels
    def read_nested_images(self, path, ftype='.png', hlevel=2):

        img_paths = []
        img_classes = []
        min_inst = np.inf
        for root, dirs, files in os.walk(path, topdown=False):
            f = [os.path.join(root, x) for x in files]

            img_paths += [x for x in f if x.endswith(ftype)]

            if len(dirs) > 0:

                # if len is 51 then is category list
                if len(dirs) == self.n_categories:
                    self.categories_label_encoder = LabelEncoder()
                    self.categories_label_encoder.fit(dirs)
                else:
                    le = LabelEncoder()
                    le.fit(dirs)
                    self.instances_label_encoders.append(le)
                    if len(dirs) < min_inst:
                        min_inst = len(dirs)

            for file in f:
                if file.endswith(ftype):
                    path = os.path.normpath(file)
                    path_components = tuple(path.split(os.sep)[:-1])
                    img_classes.append(path_components[-hlevel:])

        return img_paths, img_classes


def pop_instances(X, Y, max_instances, category_encoder, label_encoders):
    new_X = [x for x, y in zip(X, Y)
             if label_encoders[category_encoder.transform([y[0]])[0]].transform([y[1]])[0] < max_instances]

    new_Y = [y for y in Y
             if label_encoders[category_encoder.transform([y[0]])[0]].transform([y[1]])[0] < max_instances]

    return new_X, new_Y


"""
                NOT SUPPORTING TRAINING
"""
class RGBDObjects(HierarchicalImageFolder):

    def __init__(self, latent_factor_indices=None, batch_size=1, random_state=0, resize=64, center_crop=None,
                 split="train", **kwargs):

        self.split = split
        self.path = os.path.join(RGBDOBJECTS_PATH, "eval" if split == "eval" else "train")

        pad_if_needed = A.Compose([
                                    A.Lambda(name='square_pad',image=SquarePad, mask=SquarePad),
                                   A.Resize(resize, resize, p=1)], is_check_shapes=False, p=1)

        self.transform = T.Compose([T.ToTensor()])
        self.albumentation_transform = pad_if_needed

        self.data_shape = [resize, resize, 3]

        super(RGBDObjects, self).__init__(root=self.path, transforms=self.transform)

        # select just the first 3 instances per category
        self.max_instances = 1
        self.imgs, self.classes = pop_instances(self.imgs, self.classes, self.max_instances,
                                                self.categories_label_encoder, self.instances_label_encoders)

        self.factor_names = ["category", "instance", "azimuth", "pose"]
        self.factor_sizes = [self.n_categories, self.max_instances, 4, 263]

    def get_images(self, index):
        batch_imgs, classes = self.__getitem__(index)

        return classes, batch_imgs, classes

    def num_images(self):
        return len(self)

    def num_channels(self):
        return 3

    def get_shape(self):
        return self.data_shape

    @property
    def num_factors(self):
        return 4

    @property
    def factors_sizes(self):
        return [self.n_categories, self.max_instances, 4, 263]

    @property
    def full_factors_sizes(self):
        return [self.n_categories, self.max_instances, 4, 263]

    @property
    def observation_shape(self):
        return self.data_shape


class RGBDObjectsDepth(HierarchicalImageFolder):

    def __init__(self, latent_factor_indices=None, batch_size=1, random_state=0, resize=64, center_crop=None,
                 split="train", **kwargs):
        # By default, all factors are considered ground truth
        # factors.

        self.split = split
        self.path = os.path.join(RGBDOBJECTS_PATH, "eval" if split == "eval" else "train")

        pad_if_needed = A.Compose([
                                    A.Lambda(name='square_pad',image=SquarePad, mask=SquarePad),
                                   A.Resize(resize, resize, p=1)], is_check_shapes=False, p=1)

        self.transform = T.Compose([T.ToTensor()])
        self.albumentation_transform = pad_if_needed

        self.data_shape = [resize, resize, 3]

        super(RGBDObjectsDepth, self).__init__(root=self.path, transforms=self.transform, ftype="depthcrop.png")

        # select just the first 3 instances per category
        self.max_instances = 1
        self.imgs, self.classes = pop_instances(self.imgs, self.classes, self.max_instances,
                                                self.categories_label_encoder, self.instances_label_encoders)

        self.factor_names = ["category", "instance", "azimuth", "pose"]
        self.factor_sizes = [self.n_categories, self.max_instances, 4, 263]




    def __getitem__(self, index):

        img_path, (category, instance) = self.imgs[index], self.classes[index]

        category_idx = self.categories_label_encoder.transform([category])[0]
        instance_idx = self.instances_label_encoders[category_idx].transform([instance])[0]

        rr = re.findall(r'\d+', img_path)  # first is number of instance
        azimuth_idx = int(rr[-2]) - 1
        pose_idx = int(rr[-1]) - 1

        img = loader_binary(img_path)
        img = img* (255 / 4)

        img = img.astype(np.uint8)


        if self.albumentation_transform is not None:
            augmented = self.albumentation_transform(image=img)

            img = augmented['image']

        labels = np.array([category_idx, instance_idx, azimuth_idx, pose_idx])
        return self.transform(img), torch.from_numpy(labels)


    def num_channels(self):
        return 1

    def get_images(self, index):
        batch_imgs, classes = self.__getitem__(index)

        return classes, batch_imgs, classes

    def num_images(self):
        return len(self)

    def num_channels(self):
        return 3

    def get_shape(self):
        return self.data_shape

    @property
    def num_factors(self):
        return 4

    @property
    def factors_sizes(self):
        return [self.n_categories, self.max_instances, 4, 263]

    @property
    def full_factors_sizes(self):
        return [self.n_categories, self.max_instances, 4, 263]

    @property
    def observation_shape(self):
        return self.data_shape


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset = RGBDObjectsDepth()

    print(len(dataset))

    max_h = 0
    max_w = 0

    for i in range(5):

        img, label = dataset[i]

        img = img.numpy()

        _, heigth, width = img.shape

        if heigth > max_h and width > max_w:
            max_h = heigth
            max_w = width
        img = np.moveaxis(img, 0, -1)
        plt.imshow(img)
        plt.show()
        print(label)

    print(max_h, max_w)
