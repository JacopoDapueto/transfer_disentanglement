'''
Proposed disentanglement metric
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics.metric import Metric
from src.utils.postprocess_utils import *


def plot_cov_factors(X, Y, path):
    factors = X.files  # assuming X.files and Y.files corresponds
    num_points, n_features = X[factors[0]].shape

    dict_cov = {}
    for factor in factors:
        x, y = X[factor], Y[factor]

        x, y = random_samples(x, y)

        cov = np.corrcoef(x, y, rowvar=False, bias=False)

        cov = np.where(cov > 0, 1 - cov, -1 - cov)

        cov = np.abs(cov)

        # select diagonal
        cov = np.diag(cov[:n_features, n_features:])

        dict_cov[factor] = cov

    plt.figure(figsize=(8, 6), dpi=300)
    covs = np.hstack([np.expand_dims(cov, axis=1) for factor, cov in dict_cov.items()])
    sns.heatmap(covs, xticklabels=factors, annot=True, cmap="Blues", cbar=True)

    plt.xlabel("Factors")
    plt.ylabel("Dimensions")
    plt.title('Relation matrix')
    plt.savefig(os.path.join(path, "cov_heatmap.png"))
    plt.clf()


def random_samples(x, y, n=100000):

    if n > x.shape[0]:
        n= x.shape[0]
    index = np.random.choice(x.shape[0], n, replace=False)
    return  x[index], y[index]


def _overlapping_score(dim, idx_factor, covs):
    '''
    Penalize encoding overlap.
    Perfect score: 1.
    '''

    # zeros exept where factor is encoded
    perfect_cov = np.zeros(covs[dim, :].shape)
    perfect_cov[idx_factor] = 1.0

    return 1. - (np.absolute(np.subtract(perfect_cov, covs[dim, :]))).mean()


def overlapping_score(idx_factor, covs, gamma=False):
    '''
    Penalize encoding overlap.
    Consider max overlap or average
    Perfect score: 1.
    '''

    o_scores = []

    for dim in range(covs.shape[0]):
        o_score = _overlapping_score(dim, idx_factor, covs)
        o_scores.append(o_score)

    if gamma == "avg":
        # consider all dimensions in the computation
        return np.average(o_scores * covs[:, idx_factor], weights=  covs[:, idx_factor])

    # consider only best overlap
    return np.max(o_scores * covs[:, idx_factor])




def _multiple_encoding_score(dim, idx_factor, covs):
    '''
    Penalize multiple encoding.

    Perfect score: 1.
    '''


    # zeros exept where factor is encoded
    perfect_cov = np.zeros(covs[:, idx_factor].shape)
    perfect_cov[dim] = 1.0

    return 1. - np.absolute(np.subtract(perfect_cov, covs[:, idx_factor])).mean()


def multiple_encoding_score(idx_factor, covs, gamma):
    '''
    Penalize multiple encoding.
    Perfect score: 1.
    '''
    me_scores = []

    for dim in range(covs.shape[0]):
        me_score = _multiple_encoding_score(dim, idx_factor, covs)
        me_scores.append(me_score)


    me_scores = np.array(me_scores)
    if gamma == "avg":

        return np.average(me_scores * covs[:, idx_factor], weights=covs[:, idx_factor])

    # consider only best overlap
    return np.max(me_scores * covs[:, idx_factor])


def disentanglement_score(covs, factors, alpha=0.5, gamma = "avg"):
    scores = {f: 0.0 for f in factors}

    for i, factor in enumerate(factors):
        overlap = overlapping_score(i, covs, gamma)

        encoding = multiple_encoding_score(i, covs, gamma)

        factor_score = alpha * overlap + (1 - alpha) * encoding

        scores[factor] = factor_score

    return np.mean(list(scores.values())).round(4), scores

def is_inactive(representation, n_eval=10000, threshold=0.05):
    eval_representation_pos = np.random.choice(representation.shape[0], size=n_eval)

    eval_representation = representation[eval_representation_pos, :]

    variances = np.var(eval_representation, axis=0, ddof=1)

    scale_z = np.sqrt(variances)

    return scale_z < threshold


def associate_dims_factors(covs, factors):
    n_features, n_factors = covs.shape

    dict_association = {}  # factor --> list of dims


    # for each factor find dimensions with min overlap
    for i, factor in enumerate(factors):

        o_scores = []

        for dim in range(covs.shape[0]):
            o_score = _overlapping_score(dim, i, covs)
            o_scores.append(o_score)

        best = np.argmax(o_scores)

        dict_association[factor] = None

        # factor with no dimensions survived
        if np.isclose(o_scores[best], 0.0):
            continue

        dict_association[factor] = [best]
    return dict_association


def cov_factor(X, Y):
    '''
        Build "inverted" cross-covariance matrix of samples X and Y
    '''

    factors = X.files  # assuming X.files and Y.files corresponds
    num_points, n_features = X[factors[0]].shape

    dict_cov = {}

    for factor in factors:
        x, y = X[factor], Y[factor]
        x, y = random_samples(x, y)

        cov = np.corrcoef(x, y, rowvar=False, bias=False)

        cov = np.where(cov > 0, 1 - cov, -1 - cov) # 1 FoV is encoded, 0 FoV is not encoded

        cov = np.abs(cov)

        cov = cov[:n_features, n_features:]

        # interested only in the diagonal
        dict_cov[factor] = np.diag(cov.round(4))

    covs = np.hstack([np.expand_dims(cov, axis=1) for factor, cov in dict_cov.items()])

    return covs, factors


def representation_info(X, Y, representation, alpha=0.5, gamma = "avg"):
    '''
    Given X and Y representations to which an intevertion has been applied according to the factor classes.
    Get info about representation.
    '''

    covs, factors = cov_factor(X, Y)

    is_dead = is_inactive(representation)

    print("Dead dimensions [0, N): ", [i for i, dead in enumerate(is_dead) if dead])

    dict_association = associate_dims_factors(covs, factors)

    print("Association factor --> dimension(s):  ", dict_association)

    score = disentanglement_score(covs, factors, alpha, gamma)

    print("Disentanglement score: {}".format(score))



def get_score(X, Y, representation, alpha=0.5, gamma = "avg"):

    covs, factors = cov_factor(X, Y)

    inactives = is_inactive(representation)

    # remove inactive dimensions
    inactives = [i for i, inactive in enumerate(inactives) if inactive]
    covs = np.delete(covs, inactives, axis=0)

    association = associate_dims_factors(covs, factors)

    score, factors_score  = disentanglement_score(covs, factors, alpha, gamma)
    return score, inactives, association, covs, factors_score


class OMES(Metric):

    def __init__(self,  mode, **kwargs):

        super(OMES, self).__init__(**kwargs)

        self.alpha_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 1.0, 0.0]  # parameter for my metric
        self.pooling_list = ["max", "avg"]

        self.representation = np.load(self.representation_path + ".npz")[mode]

        _, file = os.path.split(self.representation_path)

        # save random couples (x, y) varing one factor if they are available
        dir, _ = os.path.split(self.classes_path)

        #mode_dir = os.path.join(dir, mode)
        self.x_path = os.path.join(dir, "X_" + file + ".npz")
        self.y_path = os.path.join(dir, "Y_" + file + ".npz")

        if not os.path.exists(dir):
            # if the demo_folder directory is not present then create it.
            os.makedirs(dir)

        if not os.path.exists(self.x_path) and not os.path.exists(self.y_path):
            X, Y = get_one_diff_couples(self.representation_path, self.classes_path, mode)

            np.savez_compressed(self.x_path, **X)
            np.savez_compressed(self.y_path, **Y)



    def get_score(self):
        ''' Return the score '''

        X = np.load(self.x_path)
        Y = np.load(self.y_path)

        alpha_scores = {}
        for pooling in self.pooling_list:

            alpha_scores[pooling] = {}
            for alpha in self.alpha_list:
                score, inactive, association, cov, _ = get_score(X, Y, self.representation, alpha=alpha, gamma=pooling)
                alpha_scores[pooling][alpha] = score

        # save association factor --> dimesions
        current_dir, _ = os.path.split(self.classes_path)
        association = pd.DataFrame.from_dict(association, orient='columns')
        association.to_csv(os.path.join(current_dir, "association.csv"), index=False)

        inactive = pd.DataFrame.from_dict({"inactive": inactive })
        inactive.to_csv(os.path.join(current_dir, "inactive.csv"), index=False)

        cov = pd.DataFrame(cov)
        cov.to_csv(os.path.join(current_dir, "cov.csv"), index=True)

        plot_cov_factors(X,Y, current_dir)

        return alpha_scores


class OMESFactors(OMES):

    def __init__(self, mode, **kwargs):

        super(OMESFactors, self).__init__(mode, **kwargs)

    def get_score(self):
        ''' Return the score '''

        X = np.load(self.x_path)
        Y = np.load(self.y_path)

        alpha_scores = {}
        for pooling in self.pooling_list:

            alpha_scores[pooling] = {}
            for alpha in self.alpha_list:
                _, _, _, _, factors_score = get_score(X, Y, self.representation, alpha=alpha, gamma=pooling)
                alpha_scores[pooling][alpha] = factors_score
        return alpha_scores
