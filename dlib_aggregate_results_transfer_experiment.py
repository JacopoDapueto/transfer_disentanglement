"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags
from configs.named_experiment import get_named_experiment

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("input_experiment", None, "Name of the experiment model to transfer from")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")

def aggregate_by_model(output_directory, state, experiment):

    # train model
    n_models = experiment.get_number_sweep()

    aggregate_results = {}

    for i in range(n_models):

        aggregate_results[i]= {}

        
        model_directory = os.path.join(output_directory, str(i), state)

        # load saved config
        with open(os.path.join(model_directory, 'info.pkl'), 'rb') as f:
            train_config = pickle.load(f)
            aggregate_results[i].update(train_config)
	
        postprocessing_config = experiment.get_postprocess_config()
        eval_config = experiment.get_eval_config()

        evaluation_directory = os.path.join(model_directory, "evaluation" )

        metrics_dict = {}
        for metric in eval_config["metrics"]:
            metrics_dict[metric] = {}

            for mode in eval_config["mode_eval"]:
		
                noises = postprocessing_config.get("label_noise")
		
                if noises is None:
                        metric_directory = os.path.join(evaluation_directory, metric, mode, "scores.pkl")

                        # load scores
                        with open(metric_directory, 'rb') as f:
                    	        scores = pickle.load(f)
                    	        metrics_dict[metric][mode] = scores
                else:
                        metrics_dict[metric][mode] = {}
                        for noise in noises:
                                metric_directory = os.path.join(evaluation_directory, metric, mode, noise, "scores.pkl")

                                # load scores
                                with open(metric_directory, 'rb') as f:
                                        scores = pickle.load(f)
                                        metrics_dict[metric][mode][noise] = scores

        # load scores
        loss_directory = os.path.join(evaluation_directory, "loss", "loss_scores.pkl")
        with open(loss_directory, 'rb') as f:
            scores = pickle.load(f)
            for k, v in scores.items():
                metrics_dict[k] = {}
                metrics_dict[k]["loss"] = v

        aggregate_results[i].update({"metrics" : metrics_dict})

    # save aggregated results
    os.makedirs(os.path.join(output_directory, state))
    with open(os.path.join(output_directory, state, 'aggregated_results.pkl'), 'wb') as handle:
        pickle.dump(aggregate_results, handle, protocol=pickle.HIGHEST_PROTOCOL)




def get_output_directory(FLAGS):

    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{input_experiment}_to_{experiment}")
    else:
        output_directory = FLAGS.output_directory

    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(experiment=str(FLAGS.experiment),
                                               input_experiment = str(FLAGS.input_experiment))

    return output_directory


def main(unused_args):
    output_directory = get_output_directory(FLAGS)

    # make experiment directory, check if experimnt folde exist
    if not os.path.exists(output_directory):
        raise FileExistsError("Experiment folder does not exist: {}".format(output_directory))


    experiment = get_named_experiment(FLAGS.experiment)

    # before transfer
    aggregate_by_model(output_directory, "before", experiment)

    # after transfer
    aggregate_by_model(output_directory, "after", experiment)


if __name__ == "__main__":
    app.run(main)
