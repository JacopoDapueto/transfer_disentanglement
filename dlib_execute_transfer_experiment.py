"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil

from absl import app
from absl import flags

from configs.named_experiment import get_named_experiment
from src.evaluation.eval_representation import evaluation_model
from src.postprocessing.postprocess import postprocess_model
from src.traininig.fine_tune import train_model
from src.visualization.simple_visualize import visualize_model as simple_visualize_model
from src.visualization.visualize import visualize_model

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("input_experiment", None, "Name of the experiment model to transfer from")
flags.DEFINE_integer("save_interval", 10000, "Number of iterations after which save the model")

flags.DEFINE_string("model_num", None, "Directory to save trained model in")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite output directory.")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")

flags.DEFINE_integer("grad_acc_steps", 1, "Number of steps of gradient accumulation")



def copytree(src, dst, symlinks=False, ignore=None):

    if not os.path.exists(dst):
        # if the demo_folder directory is not present then create it.
        os.makedirs(dst)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

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


def get_input_directory(FLAGS):
    # Set correct output directory. and check they already exist
    input_directory = os.path.join("output", "{experiment}", "{model_num}")

    # Insert model number and study name into path if necessary.
    input_directory = input_directory.format(model_num=str(FLAGS.model_num),
                                               experiment=str(FLAGS.input_experiment))

    if not os.path.exists(input_directory):
        raise FileExistsError("Input experiment folder do not exists")

    return input_directory

def main(unused_args):

    output_directory = get_output_directory(FLAGS)
    input_directory = get_input_directory(FLAGS)

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

    input_experiment = get_named_experiment(FLAGS.input_experiment)
    experiment = get_named_experiment(FLAGS.experiment)


    # BEFORE-TRANSFER
    
    pre_output_directory = os.path.join(output_directory, "before")

    # copy trained model into new experiment
    copytree(os.path.join(input_directory, "model"), os.path.join(pre_output_directory, "model"))

    # load saved config

    # save visualization of the model
    train_config = experiment.get_model_config(model_num=int(FLAGS.model_num))


    with open(os.path.join(pre_output_directory,'info.pkl'), 'wb') as handle:
        pickle.dump(train_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    config = experiment.get_postprocess_config()
    train_config.update(config)

    print("-"*20, "BEFORE", "-"*20)
    print(train_config)

   #experiment.print_postprocess_config()

    # extract and save representation
    postprocess_model(pre_output_directory, train_config)

    simple_visualize_model(pre_output_directory, train_config)

    # evaluate model
    eval_config = experiment.get_eval_config()
    experiment.print_eval_config()

    eval_config.update(train_config)
    evaluation_model(pre_output_directory, eval_config)

    

    # AFTER-TRANSFER
    print("-" * 20, "AFTER", "-" * 20)
    post_output_directory = os.path.join(output_directory, "after")
    # copy trained model into new experiment
    copytree(os.path.join(input_directory, "model"), os.path.join(post_output_directory, "model"))

    # train model
    train_config = experiment.get_model_config(model_num=int(FLAGS.model_num))
    train_config["grad_acc_steps"] = FLAGS.grad_acc_steps  # gradient accumulation parameter
    train_config["save_interval"] = FLAGS.save_interval  # save model interval parameter
    train_config["model_num"] = FLAGS.model_num

    train_config.update(experiment.get_postprocess_config())

    experiment.print_model_config(model_num=int(FLAGS.model_num))

    train_model(post_output_directory, train_config)

    # load saved config
    postprocessing_config = experiment.get_postprocess_config()
    # load saved config
    with open(os.path.join(post_output_directory, 'info.pkl'), 'rb') as f:
        train_config = pickle.load(f)
        postprocessing_config.update(train_config)

    experiment.print_postprocess_config()

    # extract and save representation
    postprocess_model(post_output_directory, postprocessing_config)

    # save all visualizations
    visualize_model(post_output_directory, postprocessing_config)

    # evaluate model
    eval_config = experiment.get_eval_config()
    experiment.print_eval_config()

    eval_config.update(train_config)

    evaluation_model(post_output_directory, eval_config)



if __name__ == "__main__":
    app.run(main)
