
"""Script to compute OMES."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import json

from absl import app
from absl import flags

from src.evaluation.metrics.omes import OMES, OMESFactors

FLAGS = flags.FLAGS
flags.DEFINE_string("representation_directory", None, "Path to the folder containing the representation and FoVs labels.")
flags.DEFINE_string("model_num", None, "Directory to save trained model in")



def main():

    representation_path = os.path.join(FLAGS.representation_directory, "representations") # path to representation without .npz extension
    classes_path = os.path.join(FLAGS.representation_directory, "classes") # path to FoVs labels without .csv extension


    mode = "mean" # representation modality

    metric_mode = OMES(mode=mode, representation_path=representation_path, classes_path=classes_path)
    dict_score = metric_mode.get_score()  # score wrt alpha

    with open(os.path.join(representation_path, 'omes.json'), 'w') as fp:
        json.dump(dict_score, fp)

    mode = "sampled"  # representation modality

    metric_mode = OMESFactors(mode=mode, representation_path=representation_path, classes_path=classes_path)
    dict_score = metric_mode.get_score()  # score wrt alpha and pooling

    with open(os.path.join(representation_path, 'omes_factors.json'), 'w') as fp:
        json.dump(dict_score, fp)




if __name__ == "__main__":
    app.run(main)