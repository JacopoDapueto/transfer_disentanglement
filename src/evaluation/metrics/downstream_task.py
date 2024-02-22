"""Downstream classification task."""

import os

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from src.methods.shared.metrics.metric import Metric
from src.methods.shared.utils.metric_utils import split_train_test


def make_predictor(name):

    if name == "gbt":
        return ensemble.GradientBoostingClassifier()
    elif name == "mlp":
        return MLPClassifier(hidden_layer_sizes=[256, 256])
    else:
        raise "Name of predictor not exists!"


def _compute_loss(x_train, y_train, x_test, y_test, predictor):
  """Compute average accuracy for train and test set."""
  num_factors = y_train.shape[0]
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = predictor
    model.fit(x_train, y_train[i, :])
    train_loss.append(np.mean(model.predict(x_train) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
  return train_loss, test_loss


def compute_downstream_task(representation, factors, predictor,perc_train, perc_test, batch_size=16):

  scores = {}
  for train_size in perc_train:
    representation = representation.T
    factors = factors.T

    mus_train, mus_test, ys_train, ys_test = split_train_test(representation, observations2=factors, train_percentage=train_size)

    predictor_model = make_predictor(predictor)

    train_err, test_err = _compute_loss(
        np.transpose(mus_train), ys_train, np.transpose(mus_test),
        ys_test, predictor_model)
    size_string = str(mus_train.shape[1])
    scores[size_string +":mean_test_accuracy"] = np.mean(test_err)
    scores[size_string + ":min_test_accuracy"] = np.min(test_err)
    for i in range(len(train_err)):
      scores[size_string + ":test_accuracy_factor_{}".format(i)] = test_err[i]

  return scores




class GBT_regressor(Metric):

    def __init__(self, mode, **kwargs):

        super(GBT_regressor, self).__init__(**kwargs)
        self.mode = mode
        self.predictor = "gbt"
        self.name = "gbt_regressor"


    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values

        scores = {}
        scores[self.name] = {}
        results = compute_downstream_task(data, classes,predictor=self.predictor, perc_train=[2/3], perc_test=[1/3])

        for k, v in results.items():
            scores[self.name][k] = v

        return scores



class MLP_regressor(GBT_regressor):

    def __init__(self, **kwargs):

        super(MLP_regressor, self).__init__(**kwargs)

        self.predictor = "mlp"
        self.name = "mlp_regressor"




class GBT_regressor_pruned(GBT_regressor):
    """
    Perform regression only on selected dimensions
    """

    def __init__(self, **kwargs):

        super(GBT_regressor_pruned, self).__init__(**kwargs)

        self.predictor = "gbt"
        self.name = "gbt_regressor_pruned"

        current_dir, _ = os.path.split(self.classes_path)

        self.inactive_dims = pd.read_csv(os.path.join(current_dir, "inactive.csv")).values
        self.pruned_dims = pd.read_csv(os.path.join(current_dir, "association.csv")).values
        self.pruned_dims = np.unique(self.pruned_dims)



    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]
        data = data[:, [i for i in range(data.shape[1]) if i not in self.inactive_dims]] # remove inactive dims

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values

        scores = {}
        scores[self.name] = {}
        results = compute_downstream_task(data[:, self.pruned_dims], classes,predictor=self.predictor, perc_train=[2/3], perc_test=[1/3])

        for k, v in results.items():
            scores[self.name][k] = v

        return scores


class MLP_regressor_pruned(GBT_regressor_pruned):

    def __init__(self, **kwargs):

        super(MLP_regressor_pruned, self).__init__(**kwargs)

        self.predictor = "mlp"
        self.name = "mlp_regressor"



