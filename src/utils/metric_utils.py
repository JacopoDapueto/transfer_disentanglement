
import numpy as  np


def split_train_test(observations, observations2=None, train_percentage=0.66):
  """Splits observations into a train and test_unsupervised set.

  Args:
    observations: Observations to split in train and test_unsupervised. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]


  if observations2 is not None:
      observations2_train = observations2[:, :num_labelled_samples_train]
      observations2_test = observations2[:, num_labelled_samples_train:]
      return observations_train, observations_test, observations2_train, observations2_test


  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test_unsupervised set."
  return observations_train, observations_test
