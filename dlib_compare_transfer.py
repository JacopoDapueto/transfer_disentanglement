"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import numpy as np
from absl import app
from absl import flags
from itertools import groupby
from operator import itemgetter
import pandas as pd



if sys.version_info.major == 3 and sys.version_info.minor >= 10:

    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

from configs.named_experiment import get_named_experiment


FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("input_experiment", None, "Name of the experiment model to transfer from")
flags.DEFINE_list("values_to_aggregate", ["beta"], "name of the parameters to show aggregated data with")
flags.DEFINE_list("fov_names", ['Category', 'Azimuth', 'Pose' ], "name of the fov to show") #['Color', 'Shape', 'Scale', 'Orientation', 'PosX', 'PosY'] ['Category', 'Azimuth', 'Pose' ] ['Floor hue', 'Wall Hue', 'Object Hue', 'Scale', 'Shape', 'Orientation'] ['Object', 'Pose', 'Orientation', 'Scale']

flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")


hyperparametrs_name = {"omes":"$\alpha$", "ccd_factors": "$\alpha$", "mig":"mig", "dci-disentanglement":"dci-disentanglement",
                       "beta_vae": "beta_vae_score", "factor_vae":"factor_vae_score", "gbt_regressor": "gbt_regressor",  "gbt_regressor_pruned": "gbt_regressor_pruned",
                       "mlp_regressor": "mlp_regressor", "mlp_regressor_pruned": "mlp_regressor_pruned", "elbo": "ELBO", "reconstruction": "Recons. loss" } # name of hyperparameters of the metrics


results_list = {"elbo": "ELBO", "reconstruction": "Recons. loss", "gbt_regressor" : "GBT10000"}

metrics_list = {"beta_vae": "BetaVAE Score",
                "omes":"Our",
                "dci-disentanglement":"DCI: Disentanglement",
                "elbo": "ELBO",
                "factor_vae":"FactorVAE Score",
                "gbt_regressor": "GBT10000",
                "mig":"MIG",
                "mlp_regressor": "MLP10000",
                "reconstruction": "Recons. loss" }

# define a function for key
def key_func( keys, dictionary):
    return itemgetter(*keys)(dictionary)


def flatten(d, parent_key=()):
    items = []
    for k, v in d.items():
        new_key = parent_key + (k,) if parent_key else (k,)
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


# aggregate results for the same hyperparameter and metric
def aggregate_models(data):
    # return dictionary key is hyperparameter, values list of metric value (one for each seed)

    metrics_dict = {}
    for key, value in data:

        info_list = list(value)
        # aggregate results
        for info in info_list:
            metrics = info["metrics"]

            flatten_dict = flatten(metrics)

            # for each (metric, hyperparameter)
            for spec, result in flatten_dict.items():

                # if first element
                if spec not in metrics_dict:
                    metrics_dict[spec] = {}

                if key not in metrics_dict[spec]:
                    metrics_dict[spec][key] = []
                metrics_dict[spec][key].extend([result])
    return metrics_dict


def aggregate_metrics(data):

    def adapted_itemgetter(elem):
        return elem[:-1]

    groupby_data = groupby(data, key= adapted_itemgetter)

    aggregated_dict = {}
    for key, v in groupby_data:
        aggregated_dict[key]  = {}
        v = list(v)

        for k in v:
            difference = k[-1]
            value = data[k]
            if  difference not in aggregated_dict[key]:
                aggregated_dict[key][difference] = {}

            aggregated_dict[key][difference].update(value)

    return aggregated_dict


def load_aggregate_results(output_directory, alpha=0.0, gamma=False):

    # attributes for loss-metrics rank correlation matrix
    loss_metrics_rank_dict = {"beta_vae": [], "omes": [], "dci-disentanglement": [], "elbo": [], "factor_vae": [],
                              "gbt_regressor": [], "mig": [], "mlp_regressor": [], "reconstruction": []}

    # attributes for loss-metrics rank correlation matrix
    metrics_violinplot_dict = {"beta_vae": [], "omes": [], "dci-disentanglement": [], "factor_vae": [], "mig": []}
    downstream_task_violinplot_dict = {"mlp_regressor": {}, "gbt_regressor": {}}

    # attributes for double violinplot
    comparison_downstream_task_violinplot_dict = {"score": [], "regressor": [], "pruned": []}
    comparison_downstream_task_fov_violinplot_dict = {"score": [], "regressor": [], "pruned": [], "fov": [],
                                                      "metric": [], "dci":[], "mig":[]}

    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        raise FileExistsError("Experiment folder does not exist")


    print("Aggregate wrt ", FLAGS.values_to_aggregate)

    # load saved config
    with open(os.path.join(output_directory, 'aggregated_results.pkl'), 'rb') as f:
        results = pickle.load(f)

    experiment = get_named_experiment(FLAGS.experiment)

    print("Aggregate wrt ", FLAGS.values_to_aggregate)

    for model_num, info in results.items():
        info["model_num"] = model_num

    result_list = [ info for model_num, info in results.items()]

    # aggregated wrt experiments info
    partial_key_func = itemgetter(*FLAGS.values_to_aggregate)
    aggregated = sorted(result_list, key= partial_key_func)

    # group according to keys
    metrics = aggregate_models(groupby(aggregated, partial_key_func))


    # group same aggregator, same spec
    for spec, values in metrics.items():

        metric = [ s for s in spec if s in hyperparametrs_name.keys()][0]

        ##### prepare data for correlation rank #####
        if metric in loss_metrics_rank_dict.keys() or metric=="mlp_regressor_pruned" or metric=="gbt_regressor_pruned":
            if metric=="omes" :
                if spec[-1]==alpha and spec[-2]==gamma: # pick only alpha==0.5
                    loss_metrics_rank_dict[metric] = [v[0] for k, v in values.items()]

                    # for each fov repeat the score metric
                    for _ in ["mlp", "gbt"]:
                        for _ in ["pruned", "not"]:
                            for _ in range(len(FLAGS.fov_names)) :
                                comparison_downstream_task_fov_violinplot_dict["metric"] += [v[0] for k, v in values.items()]

            elif metric=="mig":
                for _ in ["mlp", "gbt"]:
                    for _ in ["pruned", "not"]:
                        for _ in range(len(FLAGS.fov_names)):
                            comparison_downstream_task_fov_violinplot_dict["mig"] += [v[0] for k, v in
                                                                                         values.items()]

            elif metric=="dci-disentanglement":
                for _ in ["mlp", "gbt"]:
                    for _ in ["pruned", "not"]:
                        for _ in range(len(FLAGS.fov_names)):
                            comparison_downstream_task_fov_violinplot_dict["dci"] += [v[0] for k, v in
                                                                                         values.items()]

            elif metric=="mlp_regressor" or metric=="gbt_regressor":

                if "mean_test_accuracy" in spec[-1]: # pick only mean accuracy
                    loss_metrics_rank_dict[metric] = [v[0] for k, v in values.items()]


                    downstream_task_violinplot_dict[metric] = [v[0] for k, v in values.items()]

                    comparison_downstream_task_violinplot_dict["score"] += [v[0] for k, v in values.items()]
                    comparison_downstream_task_violinplot_dict["regressor"] += [metric for k, v in values.items()]
                    comparison_downstream_task_violinplot_dict["pruned"] += [False for k, v in values.items()]

                # add single fovs
                elif "test_accuracy_factor" in spec[-1]:
                    # dict for doubleviolinplot
                    comparison_downstream_task_fov_violinplot_dict["score"] += [v[0] for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["regressor"] += [metric for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["pruned"] += [False for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["fov"] += [spec[-1][-1] for k, v in values.items()]


            elif metric=="mlp_regressor_pruned" or metric=="gbt_regressor_pruned":

                if "mean_test_accuracy" in spec[-1]: # pick only mean accuracy

                    # dict for doubleviolinplot
                    comparison_downstream_task_violinplot_dict["score"] += [v[0] for k, v in values.items()]
                    comparison_downstream_task_violinplot_dict["regressor"] += [ metric[:-7] for k, v in values.items()]
                    comparison_downstream_task_violinplot_dict["pruned"] += [ True for k, v in values.items()]

                # add single fovs
                if "test_accuracy_factor" in spec[-1]:
                    # dict for doubleviolinplot
                    comparison_downstream_task_fov_violinplot_dict["score"] += [v[0] for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["regressor"] += [metric[:-7] for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["pruned"] += [True for k, v in values.items()]
                    comparison_downstream_task_fov_violinplot_dict["fov"] += [spec[-1][-1] for k, v in values.items()]



            elif metric=="elbo":
                loss_metrics_rank_dict[metric] = [v[0] for k, v in values.items()]

            elif metric=="reconstruction":
                loss_metrics_rank_dict[metric] = [-v[0] for k, v in values.items()]
            else:
                loss_metrics_rank_dict[metric] = [ v[0] for k,v in values.items()]


        ##### prepare data for violinplots #####
        if metric in metrics_violinplot_dict.keys():
            if metric=="omes" :
                if spec[-1]==alpha and spec[-2]==gamma: # pick only alpha==0.5

                    metrics_violinplot_dict[metric] = [v[0] for k, v in values.items()]
            else:
                metrics_violinplot_dict[metric] = [ v[0] for k,v in values.items()]



    df_comparison_downstream_task_fov_violinplot = pd.DataFrame.from_dict(comparison_downstream_task_fov_violinplot_dict)
    mlp_df = df_comparison_downstream_task_fov_violinplot.loc[
        df_comparison_downstream_task_fov_violinplot["regressor"] == "mlp_regressor"]

    gbt_df = df_comparison_downstream_task_fov_violinplot.loc[
        df_comparison_downstream_task_fov_violinplot["regressor"] == "gbt_regressor"]

    return gbt_df

def to_mean(df):

    to_latex = {}
    to_latex.update({fov_name: df.loc[df["fov"] == fov_name]["score"].mean()
                     for fov_name in df["fov"]})

    to_latex["All"] = np.mean([to_latex[fov_name] for fov_name in df["fov"]])

    # add disentanglement score
    to_latex.update({"Our disentanglement metric": np.mean(df["metric"])})
    to_latex.update({"DCI": np.mean(df["dci"])})
    to_latex.update({"MIG": np.mean(df["mig"])})

    return to_latex
def to_improvement(df):

    to_latex = {}
    to_latex.update({fov_name: df.loc[df["fov"] == fov_name]["acc_imp"].mean()
                     for fov_name in df["fov"]})

    to_latex["All"] = np.mean([to_latex[fov_name] for fov_name in df["fov"]])

    # add disentanglement score
    to_latex.update({"Our disentanglement metric": np.mean(df["dis_imp"])})
    to_latex.update({"DCI": np.mean(df["dci_imp"])})
    to_latex.update({"MIG": np.mean(df["mig_imp"])})

    return to_latex

def to_latex_table(df):
    # change labels of fov
    df.replace({"fov": {f"{i}": name for i, name in enumerate(FLAGS.fov_names)}}, inplace=True)

    pruned_ = df.loc[df["pruned"] == True]
    not_pruned_ = df.loc[df["pruned"] == False]

    # add statistics Pruned= False
    to_latex = to_improvement(not_pruned_)
    to_latex["Pruned"] = False

    df_not_pruned = pd.DataFrame(to_latex, index=[0])

    # add statistics Pruned= True
    to_latex = to_improvement(pruned_)
    to_latex["Pruned"] = True

    df_pruned = pd.DataFrame(to_latex, index=[0])

    # append rows
    df_improvement_to_latex = pd.concat([df_not_pruned, df_pruned], ignore_index=True)

    # compute means BEFORE finetuinign
    to_latex = to_mean(not_pruned_)
    to_latex["Pruned"] = False

    df_not_pruned = pd.DataFrame(to_latex, index=[0])

    to_latex = to_mean(pruned_)
    to_latex["Pruned"] = True

    df_pruned = pd.DataFrame(to_latex, index=[0])
    df_baseline_to_latex = pd.concat([df_not_pruned, df_pruned], ignore_index=True)

    df_join = join_improvement_baseline(df_baseline_to_latex, df_improvement_to_latex)

    # move pruned column
    column_pruned = df_join.pop("Pruned")
    df_join.insert(0, "Pruned", column_pruned)

    return df_join


def join_improvement_baseline(df_baseline, df_improvement):


    df_join = df_baseline.copy()

    # gives a tuple of column name and series
    # for each column in the dataframe
    for (column, data) in df_improvement.items():

        if column == "Pruned":
            continue

        data = data.map('{:+,.1f}'.format)

        # append
        df_join[column] = df_join[column] * 100.
        df_join[column] = df_join[column].map('{:,.1f}'.format) + " (" + data + ")"


    return df_join


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

    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = get_output_directory(FLAGS)
    else:
        output_directory = FLAGS.output_directory

    alpha = 0.0 # considering only the overalp_score which is the Modularity
    gamma = True

    # before transfer
    performances_before = load_aggregate_results(os.path.join(output_directory, "before"), alpha, gamma)

    # after transfer
    performances_after = load_aggregate_results(os.path.join(output_directory, "after"), alpha, gamma)

    accuracy_improvement = performances_after["score"] - performances_before["score"]
    disentanglement_improvement = performances_after["metric"] - performances_before["metric"]
    dci_improvement = performances_after["dci"] - performances_before["dci"]
    mig_improvement = performances_after["mig"] - performances_before["mig"]


    performances_before["acc_imp"] = accuracy_improvement * 100.
    performances_before["dis_imp"] = disentanglement_improvement * 100.
    performances_before["mig_imp"] = mig_improvement * 100.
    performances_before["dci_imp"] = dci_improvement * 100.

    to_latex = to_latex_table(performances_before)

    to_latex.style.hide(axis="index").to_latex(os.path.join(output_directory, "improvement_gbt.tex"))

if __name__ == "__main__":
    app.run(main)
