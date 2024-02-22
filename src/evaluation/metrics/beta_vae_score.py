"""
Mutual Information Gap from the beta-TC-VAE paper.
Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

import numpy as np
import pandas as pd
from sklearn import linear_model

from src.methods.shared.metrics.metric import Metric
from src.methods.shared.utils.metric_utils import split_train_test


def _get_one_shared_couples(representation, factors, index, batch_size, random_state):

    points1= np.zeros((batch_size, representation.shape[1]))
    points2= np.zeros((batch_size, representation.shape[1]))
    for i in range(batch_size):

        sample_pos = random_state.randint(representation.shape[0])
        point1 = representation[sample_pos, :]
        f = factors[sample_pos, index]

        # Ensure sampled coordinate is the same across pairs of samples.
        candidates = np.where(factors[:, index] == f)[0]
        sample_pos = candidates[random_state.randint(candidates.shape[0])]
        point2 = representation[sample_pos, :]

        points1[i, :] = point1
        points2[i, :] = point2

    return points1, points2


def _generate_training_sample(representation, factors, batch_size, random_state):

    # Select random coordinate to keep fixed.
    index = random_state.randint(factors.shape[1])

    # Sample two mini batches of latent variables.
    representation1, representation2 = _get_one_shared_couples(representation, factors, index, batch_size, random_state)


    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
    return index, feature_vector


def _generate_training_batch(representation, factors, batch_size, num_points, random_state):

    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = _generate_training_sample(representation, factors, batch_size, random_state)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def compute_beta_vae_sklearn(representation, factors, random_state, perc_train, perc_test, batch_size=64):

  representation = representation.T
  factors = factors.T

  mus_train, mus_test, ys_train, ys_test = split_train_test(representation, observations2=factors, train_percentage=perc_train)

  mus_train, mus_test = mus_train.T, mus_test.T
  ys_train, ys_test = ys_train.T, ys_test.T

  train_points, train_labels = _generate_training_batch(mus_train, ys_train, batch_size, 10000, random_state)
  model = linear_model.LogisticRegression(random_state=random_state)
  model.fit(train_points, train_labels)

  eval_points, eval_labels = _generate_training_batch(mus_test, ys_test, batch_size, 5000, random_state)

  eval_accuracy = model.score(eval_points, eval_labels)


  return eval_accuracy



class BetaVaeScore(Metric):

    def __init__(self, mode, **kwargs):
        super(BetaVaeScore, self).__init__(**kwargs)
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

        scores["beta_vae_score"]=compute_beta_vae_sklearn(data, classes, self.random_state, perc_train=2/3, perc_test=1/3)
        return scores

