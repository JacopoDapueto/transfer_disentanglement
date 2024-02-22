"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil

from absl import app
from absl import flags
from src.config.named_experiment import get_named_experiment
from src.methods.shared.evaluation.eval_representation import evaluation_model
from src.methods.shared.postprocessing.postprocess import postprocess_model
from src.methods.shared.traininig.train import train_model
from src.methods.shared.visualization.visualize import visualize_model

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("model_num", None, "Directory to save trained model in")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite output directory.")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")
flags.DEFINE_integer("grad_acc_steps", 1, "Number of steps of gradient accumulation")



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
        # if the demo_folder directory is not present then create it.
        os.makedirs(output_directory)
    else:
        if FLAGS.overwrite:
            shutil.rmtree(output_directory) # remove project
            os.makedirs(output_directory) # recreate it
        else:
            raise FileExistsError("Experiment folder exists")

    experiment = get_named_experiment(FLAGS.experiment)

    # train model
    train_config = experiment.get_model_config(model_num=int(FLAGS.model_num))
    train_config["grad_acc_steps"] = FLAGS.grad_acc_steps  # gradient accumulation parameter
    experiment.print_model_config(model_num=int(FLAGS.model_num))
    train_model(output_directory, train_config)

    # save visualization of the model
    postprocessing_config = experiment.get_postprocess_config()
    # load saved config
    with open(os.path.join(output_directory, 'info.pkl'), 'rb') as f:
        train_config = pickle.load(f)
        postprocessing_config.update(train_config)

    experiment.print_postprocess_config()

    # extract and save representation
    postprocess_model(output_directory, postprocessing_config)

    # save all visualizations
    visualize_model(output_directory, postprocessing_config)

    # evaluate model
    eval_config = experiment.get_eval_config()
    experiment.print_eval_config()
    eval_config.update(train_config)
    evaluation_model(output_directory, eval_config)


if __name__ == "__main__":
    app.run(main)
