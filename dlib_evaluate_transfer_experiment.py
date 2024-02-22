"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags
from src.config.named_experiment import get_named_experiment
from src.methods.shared.evaluation.eval_representation import evaluation_model
from src.methods.shared.postprocessing.postprocess import postprocess_model
from src.methods.shared.visualization.visualize import visualize_model

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("input_experiment", None, "Name of the experiment model to transfer from")

flags.DEFINE_string("model_num", None, "Directory to save trained model in")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite output directory.")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")


flags.DEFINE_boolean("visualize", False, "Create visualizations")
flags.DEFINE_boolean("postprocess", False, "Create posprocessing")

def get_output_directory(FLAGS):

    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{input_experiment}_to_{experiment}", "{model_num}")
    else:
        output_directory = FLAGS.output_directory

    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=str(FLAGS.model_num),
                                               experiment=str(FLAGS.experiment),
                                               input_experiment = str(FLAGS.input_experiment))

    return output_directory



def main(unused_args):

    output_directory = get_output_directory(FLAGS)

    # make experiment directory, check if experimnt folde exist
    if not os.path.exists(output_directory):
        raise FileExistsError("Experiment folder exists")

    input_experiment = get_named_experiment(FLAGS.experiment)
    experiment = get_named_experiment(FLAGS.experiment)


    # BEFORE-TRANSFER
    
    pre_output_directory = os.path.join(output_directory, "before")

    # save visualization of the model
    postprocessing_config = experiment.get_postprocess_config()

    with open(os.path.join(pre_output_directory, 'info.pkl'), 'rb') as f:
        train_config = pickle.load(f)
        train_config.update(postprocessing_config)
        postprocessing_config = train_config

    # evaluate model, postprocessing already done
    eval_config = experiment.get_eval_config()

    if FLAGS.postprocess:
        # extract and save representation
        postprocess_model(pre_output_directory, postprocessing_config)

    train_config.update(eval_config)
    eval_config = train_config
    experiment.print_eval_config()
    evaluation_model(pre_output_directory, eval_config)


    # AFTER-TRANSFER

    post_output_directory = os.path.join(output_directory, "after")

    postprocessing_config = experiment.get_postprocess_config()
    with open(os.path.join(post_output_directory, 'info.pkl'), 'rb') as f:
        train_config = pickle.load(f)
        train_config.update(postprocessing_config)
        postprocessing_config = train_config


    experiment.print_postprocess_config()

    if FLAGS.postprocess:
        # extract and save representation
        postprocess_model(post_output_directory, postprocessing_config)

    # save all visualizations

    if FLAGS.visualize:
        visualize_model(post_output_directory, postprocessing_config)


    # evaluate model, postprocessing already done
    eval_config = experiment.get_eval_config()

    train_config.update(eval_config)
    eval_config = train_config

    experiment.print_eval_config()
    evaluation_model(post_output_directory, eval_config)



if __name__ == "__main__":
    app.run(main)
