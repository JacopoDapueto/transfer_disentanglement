"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags
from configs.named_experiment import get_named_experiment
from src.evaluation.eval_representation import evaluation_model

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("model_num", "0", "Directory to save trained model in")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")

def main(unused_args):

    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{experiment}", "{model_num}")
    else:
        output_directory = FLAGS.output_directory

    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=str(FLAGS.model_num),
                                               experiment=str(FLAGS.experiment))
    # make experiment directory
    if not os.path.exists(output_directory):
        
       raise FileExistsError("Experiment not folder exists")

    experiment = get_named_experiment(FLAGS.experiment)

    # save visualization of the model
    postprocessing_config = experiment.get_postprocess_config()
    # load saved config
    with open(os.path.join(output_directory, 'info.pkl'), 'rb') as f:
        train_config = pickle.load(f)
        postprocessing_config.update(train_config)

    experiment.print_postprocess_config()



    # evaluate model
    eval_config = experiment.get_eval_config()
    experiment.print_eval_config()
    evaluation_model(output_directory, eval_config)


if __name__ == "__main__":
    app.run(main)
