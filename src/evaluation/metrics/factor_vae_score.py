"""
Mutual Information Gap from the beta-TC-VAE paper.
Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics.metric import Metric
from src.utils.metric_utils import split_train_test


def _prune_dims(variances, threshold=0.05):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(representation, batch_size, random_state):
    eval_representation_pos = random_state.choice(representation.shape[0], size=batch_size)

    eval_representation = representation[eval_representation_pos, :]
    assert eval_representation.shape[0] == batch_size
    return np.var(eval_representation, axis=0, ddof=1)


def _get_same_factors(representation, factors, index, batch_size, random_state):
    points = np.zeros((batch_size, representation.shape[1]))

    sample_pos = random_state.randint(representation.shape[0])
    f = factors[sample_pos, index]

    for i in range(batch_size):
        # Ensure sampled coordinate is the same across pairs of samples.
        candidates = np.where(factors[:, index] == f)[0]
        sample_pos = candidates[random_state.randint(candidates.shape[0])]
        point2 = representation[sample_pos, :]

        points[i, :] = point2

    return points


def _generate_training_sample(representation, factors, batch_size, random_state, global_variances, active_dims):
    # Select random coordinate to keep fixed.
    index = random_state.randint(factors.shape[1])

    # Sample two mini batches of latent variables.
    modified_representation = _get_same_factors(representation, factors, index, batch_size, random_state)
    local_variances = np.var(modified_representation, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])

    return index, argmin


def _generate_training_batch(representation, factors, batch_size, num_points, random_state, global_variances,
                             active_dims):
    votes = np.zeros((factors.shape[1], global_variances.shape[0]), dtype=np.int64)

    for i in range(num_points):
        factor_index, argmin = _generate_training_sample(representation, factors, batch_size, random_state,
                                                         global_variances, active_dims)
        votes[factor_index, argmin] += 1
    return votes


def compute_factor_vae_sklearn(representation, factors, random_state, perc_train, perc_test, batch_size=64,
                               num_variance_estimate=10000):
    representation = representation.T
    factors = factors.T

    mus_train, mus_test, ys_train, ys_test = split_train_test(representation, observations2=factors, train_percentage=perc_train)


    mus_train, mus_test = mus_train.T, mus_test.T
    ys_train, ys_test = ys_train.T, ys_test.T

    global_variances = _compute_variances(representation.T, num_variance_estimate, random_state)
    active_dims = _prune_dims(global_variances)
    scores_dict = {}

    if not active_dims.any():
        scores_dict["train_accuracy"] = 0.
        scores_dict["eval_accuracy"] = 0.
        scores_dict["num_active_dims"] = 0
        return scores_dict

    training_votes = _generate_training_batch(mus_train, ys_train, batch_size, 10000, random_state, global_variances,
                                              active_dims)

    classifier = np.argmax(training_votes, axis=0)
    other_index = np.arange(training_votes.shape[1])

    train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

    eval_votes = _generate_training_batch(mus_test, ys_test, batch_size, 5000, random_state, global_variances,
                                          active_dims)

    eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)

    scores_dict = {}
    #scores_dict["train_accuracy"] = train_accuracy
    #scores_dict["eval_accuracy"] = eval_accuracy
    #scores_dict["num_active_dims"] = len(active_dims)

    return eval_accuracy


class FactorVaeScore(Metric):

    def __init__(self, mode, **kwargs):
        super(FactorVaeScore, self).__init__(**kwargs)
        self.mode = mode
        self.random_state = np.random.RandomState(0)

    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values
        scores = {}
        scores["factor_vae_score"] = compute_factor_vae_sklearn(data, classes, self.random_state, perc_train=2 / 3,
                                                              perc_test=1 / 3)
        return scores
