"""Script to run the training protocol."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
from itertools import groupby
from operator import itemgetter

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
import seaborn as sns


if sys.version_info.major == 3 and sys.version_info.minor >= 10:

    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

from src.config.named_experiment import get_named_experiment
from src.methods.shared.utils.visualize_utils import plot_metric_group, plot_metric


FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", None, "Name of the experiment to run")
flags.DEFINE_string("input_experiment", None, "Name of the experiment model to transfer from")
flags.DEFINE_list("values_to_aggregate", ["beta"], "name of the parameters to show aggregated data with")
flags.DEFINE_list("fov_names", ['Color', 'Shape', 'Scale', 'Orientation', 'PosX', 'PosY'], "name of the fov to show") #['Color', 'Shape', 'Scale', 'Orientation', 'PosX', 'PosY'] ['Category', 'Azimuth', 'Pose' ] ['Floor hue', 'Wall Hue', 'Object Hue', 'Scale', 'Shape', 'Orientation']

flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")
flags.DEFINE_boolean("before", True, "make plot of before of after transfer")


hyperparametrs_name = {"ccd":"$\alpha$", "ccd_factors": "$\alpha$", "mig":"mig", "dci-disentanglement":"dci-disentanglement",
                       "beta_vae": "beta_vae_score", "factor_vae":"factor_vae_score", "gbt_regressor": "gbt_regressor",  "gbt_regressor_pruned": "gbt_regressor_pruned",
                       "mlp_regressor": "mlp_regressor", "mlp_regressor_pruned": "mlp_regressor_pruned", "elbo": "ELBO", "reconstruction": "Recons. loss" } # name of hyperparameters of the metrics
#metric_plot_func = {"ccd": plot_metric, "ccd_factors": plot_metric_group} # kind of plot for each different metric


results_list = {"elbo": "ELBO", "reconstruction": "Recons. loss", "gbt_regressor" : "GBT10000"}

metrics_list = {"beta_vae": "BetaVAE Score",
                "ccd":"Our",
                "dci-disentanglement":"DCI: Disentanglement",
                "elbo": "ELBO",
                "factor_vae":"FactorVAE Score",
                "gbt_regressor": "GBT10000",
                "mig":"MIG",
                "mlp_regressor": "MLP10000",
                "reconstruction": "Recons. loss" }





def get_coolwarm_cmap(N):

    coolwarm_cmap = cm.get_cmap('coolwarm', N)

    return coolwarm_cmap


def make_rank_correlation_metrics_plots(data, title, path):
    N = 7

    #data = data.round(2)

    metrics = data[metrics_list.keys()]

    #metrics = metrics.round(3)

    corr = metrics.corr(method='spearman').loc[ ["elbo", "gbt_regressor", "mlp_regressor", "reconstruction"], ["beta_vae", "factor_vae", "ccd", "dci-disentanglement", "mig"]]
    #print(corr)
    #res, pvalue = stats.spearmanr(metrics.values)

    # Get the coolwarm colormap with N values
    coolwarm_cmap = get_coolwarm_cmap(N)

    fig = plt.figure()

    ax = sns.heatmap(corr * 100., cmap=coolwarm_cmap, square=True, annot=True, fmt=',.0f',
                vmax=100., vmin=-100, linewidths=0.5, annot_kws={'size': 'large'}, xticklabels=["BetaVAE Score", "FactorVAE Score", "Our", "DCI", "MIG"],
                 yticklabels=["ELBO", "GBT10000", "MLP10000", "Recons. loss"]
                , cbar=False)

    sns.set(font_scale=1.75)  # font size 2
    plt.title(title)
    plt.yticks(rotation=0)

    fig.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()



def make_metrics_violinplot(df, title, path):

    # change labels of fov
    df = df.rename(columns=metrics_list, inplace=False)

    df = df.round(3)
    sns.set_style("darkgrid")
    sns.set(font_scale=1)  # font size 2
    ax = sns.violinplot(data=df, scale="area", scale_hue=False, cut=0, xticklabels=["BetaVAE Score", "Our", "DCI", "FactorVAE Score",  "MIG"]) #
    sns.set(font_scale=1)  # font size 2

    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    #plt.title(title)
    #plt.xlabel('Factors')
    plt.ylabel("Disentanglement score")

    plt.ylim([-0.0005, 1.005])

    plt.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()  # clear figure


def make_metrics_violinplots_with_examples(df, points, title, path):
    df = df.round(3)
    sns.set_style("darkgrid")
    ax = sns.violinplot(data=df, scale="area", scale_hue=False, cut=0,  color="0.8", xticklabels=["BetaVAE Score", "Our", "DCI", "FactorVAE Score",  "MIG"])  #

    points = pd.melt(points.reset_index(), var_name='groups', value_name='vals', id_vars='index')
    sns.stripplot(data=points, x="groups", y="vals", jitter=False, edgecolor="gray", linewidth=1, hue="index",dodge=True)
    sns.set(font_scale=1)  # font size 2

    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    # plt.title(title)
    # plt.xlabel('Factors')
    plt.ylabel("Disentanglement score")

    plt.ylim([-0.0005, 1.005])

    plt.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()  # clear figure



def make_downstreamtasks_withpruned_violinplot(df, title, path, x = "regressor"):

    df = df.round(3)
    sns.set_style("darkgrid")
    ax = sns.violinplot(data=df, x=x, y="score", hue="pruned", scale="area", scale_hue=False, cut=0) #
    sns.set(font_scale=1)  # font size 2

    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    #plt.title(title)
    #plt.xlabel('Factors')
    plt.ylabel("Downstream task performances")

    plt.ylim([-0.0005, 1.005])

    plt.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()  # clear figure



def make_downstreamtasks_fovs_withpruned_violinplot(df, title, path):

    df = df.round(3)

    # change labels of fov
    df.replace({"fov": {f"{i}": name for i, name in enumerate(FLAGS.fov_names)}}, inplace=True)

    sns.set_style("darkgrid")
    ax = sns.violinplot(data=df, x="fov", y="score", hue="pruned", scale="count", scale_hue=False, cut=0) #
    sns.set(font_scale=1)  # font size 2

    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    #plt.title(title)
    #plt.xlabel('Factors')
    plt.ylabel("Downstream task performances")

    plt.ylim([-0.0005, 1.005])

    plt.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()  # clear figure



def make_transfer_score_violinplot(df, title, path):
    sns.set_style("darkgrid")

    df = df.loc[df['pruned'] == False]

    # change labels of fov
    df.replace({"fov": {f"{i}": name for i, name in enumerate(FLAGS.fov_names)}}, inplace=True)

    perc =np.percentile(df['metric'], 30)
    perc = round(perc, 2)

    df["Our score"] = np.where(df['metric']>perc, f'> {perc}', f'< {perc}')
    ax = sns.violinplot(data=df, x="fov", y="score", hue="Our score", scale="count", scale_hue=False, cut=0) # scale="count" , density_norm= "count"
    sns.set(font_scale=1)  # font size 2

    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    plt.ylim([-0.0005, 1.005])

    # sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    # plt.title(title)
    #plt.xlabel('Factors')
    plt.ylabel("Downstream task performances")

    # saving loss plot
    plt.savefig(path, dpi=1200, bbox_inches='tight', format='png')
    plt.clf()  # clear figure





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
        #print(key)

        info_list = list(value)
        # aggregate results
        for info in info_list:
            metrics = info["metrics"]

            flatten_dict = flatten(metrics)
            #specs, results = flatten_dict.keys(), flatten_dict

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
        #print(key, list(v))
        aggregated_dict[key]  = {}
        v = list(v)

        for k in v:
            difference = k[-1]
            value = data[k]
            if  difference not in aggregated_dict[key]:
                aggregated_dict[key][difference] = {}

            aggregated_dict[key][difference].update(value)

    return aggregated_dict

def plots(output_directory, alpha=0.0, gamma=False):


    # attributes for loss-metrics rank correlation matrix
    loss_metrics_rank_dict = {"beta_vae": [], "ccd": [], "dci-disentanglement": [], "elbo": [], "factor_vae": [],
                              "gbt_regressor": [], "mig": [], "mlp_regressor": [], "reconstruction": []}

    # attributes for loss-metrics rank correlation matrix
    metrics_violinplot_dict = {"beta_vae": [], "ccd": [], "dci-disentanglement": [], "factor_vae": [], "mig": []}
    downstream_task_violinplot_dict = {"mlp_regressor": {}, "gbt_regressor": {}}

    # attributes for double violinplot
    comparison_downstream_task_violinplot_dict = {"score": [], "regressor": [], "pruned": []}
    comparison_downstream_task_fov_violinplot_dict = {"score": [], "regressor": [], "pruned": [], "fov": [],
                                                      "metric": []}

    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        raise FileExistsError("Experiment folder does not exist")




    # make plot directory
    plot_dir = os.path.join(output_directory, "plot")
    if not os.path.exists(plot_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(plot_dir)

    gamma_dir = os.path.join(output_directory, "gamma_" + str(gamma))
    if not os.path.exists(gamma_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(gamma_dir)

    alpha_dir = os.path.join(gamma_dir, "alpha_" + str(alpha))
    if not os.path.exists(alpha_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(alpha_dir)


    print("Aggregate wrt ", FLAGS.values_to_aggregate)

    # load saved config
    with open(os.path.join(output_directory, 'aggregated_results.pkl'), 'rb') as f:
        results = pickle.load(f)




    experiment = get_named_experiment(FLAGS.experiment)

    print("Aggregate wrt ", FLAGS.values_to_aggregate)

    for model_num, info in results.items():
        info["model_num"] = model_num

    result_list = [ info for model_num, info in results.items()]
    #print(result_list)

    # aggregated wrt experiments info
    partial_key_func = itemgetter(*FLAGS.values_to_aggregate)
    aggregated = sorted(result_list, key= partial_key_func)

    # group according to keys
    metrics = aggregate_models(groupby(aggregated, partial_key_func))


    # group same aggregator, same spec
    for spec, values in metrics.items():
        print(spec)
        metric_name = list(spec)[0]
        metric_plot_dir = os.path.join(plot_dir, metric_name)
        if not os.path.exists(metric_plot_dir):
            # if the demo_folder directory is not present then create it.
            os.makedirs(metric_plot_dir)

        metric = [ s for s in spec if s in hyperparametrs_name.keys()][0]
        hyper_name  = hyperparametrs_name[metric]
        #plot_func = metric_plot_func[metric]

        ##### prepare data for correlation rank #####
        if metric in loss_metrics_rank_dict.keys() or metric=="mlp_regressor_pruned" or metric=="gbt_regressor_pruned":
            if metric=="ccd" :
                if spec[-1]==alpha and spec[-2]==gamma: # pick only alpha==0.5
                    loss_metrics_rank_dict[metric] = [v[0] for k, v in values.items()]

                    # for each fov repeat the score metric
                    #for k, v in values.items():
                    for _ in ["mlp", "gbt"]:
                        for _ in ["pruned", "not"]:
                            for _ in range(len(FLAGS.fov_names)) :
                                comparison_downstream_task_fov_violinplot_dict["metric"] += [v[0] for k, v in values.items()]


            elif metric=="mlp_regressor" or metric=="gbt_regressor":

                if "mean_test_accuracy" in spec[-1]: # pick only mean accuracy
                    #print(values)
                    loss_metrics_rank_dict[metric] = [v[0] for k, v in values.items()]


                    downstream_task_violinplot_dict[metric] = [v[0] for k, v in values.items()]

                                        # dict for doubleviolinplot
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
            if metric=="ccd" :
                if spec[-1]==alpha and spec[-2]==gamma: # pick only alpha==0.5

                    metrics_violinplot_dict[metric] = [v[0] for k, v in values.items()]
            else:
                metrics_violinplot_dict[metric] = [ v[0] for k,v in values.items()]

        # set common visualize parameters
        visualize_params = { "aggregated_data": values, "title": "{}".format(metric_name ), "set_lim": False}

        # plot separate metrics
        plot_metric(**visualize_params)
        plt.legend()
        plt.savefig(os.path.join(metric_plot_dir, "{}.png".format(spec)), dpi=300)
        plt.clf()

    # aggregate as much as possible by key and plot together
    agg_metrics = aggregate_metrics(metrics)

    # make plot directory
    plot_dir = os.path.join(output_directory, "plot_aggregated")
    if not os.path.exists(plot_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(plot_dir)

    for spec, values in agg_metrics.items():
        metric_name = list(spec)[0]
        metric_plot_dir = os.path.join(plot_dir, metric_name)
        if not os.path.exists(metric_plot_dir):
            # if the demo_folder directory is not present then create it.
            os.makedirs(metric_plot_dir)

        metric = [ s for s in spec if s in hyperparametrs_name.keys()]

        metric = metric[0]
        hyper_name  = hyperparametrs_name[metric]
        #plot_func = metric_plot_func[metric]

        # set common visualize parameters
        visualize_params = { "aggregated_data": values, "group":values.keys(), "title": "{}".format(spec), "set_lim": False}

        # plot separate metrics
        plot_metric_group(**visualize_params)
        plt.legend()
        plt.savefig(os.path.join(metric_plot_dir, "{}_aggregated.png".format(spec)), dpi=300)
        plt.clf()


    # plot correlations rank

    df_correlation = pd.DataFrame.from_dict(loss_metrics_rank_dict)

    make_rank_correlation_metrics_plots(df_correlation,"", os.path.join(alpha_dir, "loss_metrics_correlation.png"))

    # plot violinplots
    df_metrics_violinplot = pd.DataFrame.from_dict(metrics_violinplot_dict)
    make_metrics_violinplot(df_metrics_violinplot, "",
                                        os.path.join(alpha_dir, "metrics_violinplots.png"))

    points = df_metrics_violinplot.iloc[[0, 2, 3, 6, 14, 11, 19]]
    make_metrics_violinplots_with_examples(df_metrics_violinplot, points, "",
                            os.path.join(alpha_dir, "metrics_violinplots_with_examples.png"))


    df_downstream_task_violinplot = pd.DataFrame.from_dict(downstream_task_violinplot_dict)
    make_metrics_violinplot(df_downstream_task_violinplot, "",
                            os.path.join(alpha_dir, "downstream_task_notpruned_violinplots.png"))

    df_comparison_downstream_task_violinplot = pd.DataFrame.from_dict(comparison_downstream_task_violinplot_dict)

    make_downstreamtasks_withpruned_violinplot(df_comparison_downstream_task_violinplot, "",
                            os.path.join(alpha_dir, "downstream_task_withpruned_violinplots.png"))


    df_comparison_downstream_task_fov_violinplot = pd.DataFrame.from_dict(comparison_downstream_task_fov_violinplot_dict)
    mlp_df = df_comparison_downstream_task_fov_violinplot.loc[df_comparison_downstream_task_fov_violinplot["regressor"] == "mlp_regressor"]
    make_downstreamtasks_fovs_withpruned_violinplot(mlp_df, "",
                                               os.path.join(alpha_dir, "downstream_task_mlp_fov_violinplots.png"))

    gbt_df = df_comparison_downstream_task_fov_violinplot.loc[
        df_comparison_downstream_task_fov_violinplot["regressor"] == "gbt_regressor"]
    make_downstreamtasks_fovs_withpruned_violinplot(gbt_df, "",
                                               os.path.join(alpha_dir, "downstream_task_gbt_fov_violinplots.png"))

    make_transfer_score_violinplot(gbt_df, "", os.path.join(alpha_dir, "transferscore_gbt_fov_violinplots.png"))

    #df_comparison_downstream_task_violinplot = pd.DataFrame.from_dict(comparison_downstream_task_nocolor_violinplot_dict)
    #make_downstreamtasks_withpruned_violinplot(df_comparison_downstream_task_violinplot, "",
    #                                           os.path.join(output_directory,
    #                                                        "downstream_task_withpruned_nocolor_violinplots.png"))



    ###################
    ### PLOT X BETA ###
    ###################

    beta1_metrics_violinplot_dict= {"beta_vae": [], "ccd":[] , "dci-disentanglement":[] , "factor_vae":[], "mig":[]}
    beta2_metrics_violinplot_dict= {"beta_vae": [], "ccd":[] , "dci-disentanglement":[] , "factor_vae":[], "mig":[]}

    # load saved config
    with open(os.path.join(output_directory, 'aggregated_results.pkl'), 'rb') as f:
        results = pickle.load(f)


    for model_num, info in results.items():
        info["model_num"] = model_num
        info["beta"] = 1.0 if model_num < 10 else 2.0

    # remove model number
    result_list = [info for model_num, info in results.items()]

    # aggregated wrt experiments info
    partial_key_func = itemgetter("beta")
    aggregated = sorted(result_list, key=partial_key_func)

    # group according to keys
    metrics = aggregate_models(groupby(aggregated, partial_key_func))


    # group same aggregator, same spec
    for spec, values in metrics.items():
        print(spec)
        metric_name = list(spec)[0]
        metric_plot_dir = os.path.join(plot_dir, metric_name)
        if not os.path.exists(metric_plot_dir):
            # if the demo_folder directory is not present then create it.
            os.makedirs(metric_plot_dir)

        metric = [s for s in spec if s in hyperparametrs_name.keys()][0]
        hyper_name = hyperparametrs_name[metric]
        # plot_func = metric_plot_func[metric]



        ##### prepare data for violinplots #####
        if metric in metrics_violinplot_dict.keys():
            if metric == "ccd":
                if spec[-1] == alpha and spec[-2]==gamma:  # pick only alpha==0.5

                    beta1_metrics_violinplot_dict[metric] = values[1.0]
                    beta2_metrics_violinplot_dict[metric] = values[2.0]
            else:
                beta1_metrics_violinplot_dict[metric] = values[1.0]
                beta2_metrics_violinplot_dict[metric] = values[2.0]
        # set common visualize parameters
        visualize_params = {"aggregated_data": values, "title": "{}".format(metric_name), "set_lim": False}

        # plot separate metrics
        plot_metric(**visualize_params)
        plt.legend()
        plt.savefig(os.path.join(metric_plot_dir, "{}.png".format(spec)), dpi=300)
        plt.clf()


    # plot violinplots
    beta1_metrics_violinplot = pd.DataFrame.from_dict(beta1_metrics_violinplot_dict)
    make_metrics_violinplot(beta1_metrics_violinplot, "",
                                        os.path.join(alpha_dir, "metrics_violinplots_beta1.png"))
    beta2_metrics_violinplot = pd.DataFrame.from_dict(beta2_metrics_violinplot_dict)
    make_metrics_violinplot(beta2_metrics_violinplot, "",
                            os.path.join(alpha_dir, "metrics_violinplots_beta2.png"))

    #points = df_metrics_violinplot.iloc[[0, 2, 3, 6, 14, 11, 19]]
    #make_metrics_violinplots_with_examples(df_metrics_violinplot, points, "",os.path.join(output_directory, "metrics_violinplots_with_examples.png"))


    #df_downstream_task_violinplot = pd.DataFrame.from_dict(downstream_task_violinplot_dict)
    #make_metrics_violinplot(df_downstream_task_violinplot, "", os.path.join(output_directory, "downstream_task_violinplots.png"))




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


    for alpha in [1.0, 0.5, 0.0]:
        for gamma in [True, False]:
            # before transfer
            plots(os.path.join(output_directory, "before"), alpha, gamma)

            # after transfer
            plots(os.path.join(output_directory, "after"), alpha, gamma)


if __name__ == "__main__":
    app.run(main)
