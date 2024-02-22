from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Metric(object):
    """Abstract class for metric."""

    def __init__(self, representation_path, classes_path, **kwargs):

        super(Metric, self).__init__()

        self.representation_path = representation_path
        self.classes_path = classes_path


    def get_score(self):
        ''' Return the score '''
        raise NotImplementedError()