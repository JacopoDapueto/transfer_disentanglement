"""
Mutual Information Gap from the beta-TC-VAE paper.
Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from src.evaluation.metrics.metric import Metric


def histogram_discretizer(x, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(x)
    for i in range(x.shape[1]):
        discretized[:, i] = np.digitize(x[:, i], np.histogram(x[:, i], num_bins)[1][:-1])
    return discretized


def discrete_entropy(x):
    """Compute discrete mutual information."""
    num_factors = x.shape[1]
    return np.array([ mutual_info_score(x[:, j], x[:, j]) for j in range(num_factors) ])


def discrete_mutual_info(x, y):
    """Compute discrete mutual information."""
    num_codes = x.shape[1]
    num_factors = y.shape[1]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(y[:, j], x[:, i])
    return m


def discrete_mig(representation, gt_factors):
    """Computes score on test_unsupervised data."""

    discretized_rep = histogram_discretizer(representation)
    dmi = discrete_mutual_info(discretized_rep, gt_factors)

    assert dmi.shape[0] == representation.shape[1]
    assert dmi.shape[1] == gt_factors.shape[1]

    # dmi is [num_latents, num_factors]
    entropy = discrete_entropy(gt_factors)
    sorted_m = np.sort(dmi, axis=0)[::-1]

    #print(sorted_m, entropy)

    return np.mean( np.divide(sorted_m[0, :] - sorted_m[1, :], entropy, out=np.zeros_like(sorted_m[0, :] - sorted_m[1, :]), where=entropy!=0))


class MIG(Metric):

    def __init__(self, mode, **kwargs):

        super(MIG, self).__init__(**kwargs)
        self.mode = mode

    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values
        scores = {}
        scores["mig"] = discrete_mig(data, classes)
        return scores


class MIGFactors(MIG):

    def __init__(self, **kwargs):

        super(MIGFactors, self).__init__(**kwargs)

    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values

        latents_names = csv.columns

        scores = {}
        scores["mig"] = {}
        for i, name in enumerate(latents_names):
            scores["mig"][name] = discrete_mig(data, np.expand_dims(classes[:, i], axis=1))
        return scores
