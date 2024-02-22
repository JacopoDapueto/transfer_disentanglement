
"""Abstract base class for an experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Experiment(object):
  """Abstract base class for experiments."""

  def __init__(self):
    
    super(Experiment, self).__init__()
    self.model_config = None
    self.postprocess_config = None
    self.eval_config = None

  def get_number_sweep(self):
    raise NotImplementedError()

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    raise NotImplementedError()

  def print_model_config(self, model_num=0):
    """Prints model bindings and config file."""
    model_config_file = self.get_model_config(model_num)
    print("Gin base config for model training:")
    print("--")
    print(model_config_file)
    print()
    print("--")

  def get_postprocess_config(self):
    """Returns postprocessing config."""
    raise NotImplementedError()

  def print_postprocess_config(self):
    """Prints postprocessing config files."""
    print("Gin config files for postprocessing:")
    print("--")
    print(self.get_postprocess_config())


  def get_eval_config(self):
    """Returns evaluation config files."""
    raise NotImplementedError()

  def print_eval_config(self):
    """Prints evaluation config files."""
    print("Gin config files for evaluation:")
    print("--")
    print(self.get_eval_config())
