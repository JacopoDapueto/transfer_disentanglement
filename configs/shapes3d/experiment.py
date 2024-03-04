# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hyperparameter sweeps and configs for the Autoencoder experiment
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import configs.hyperparameters as h
from configs.experiment import Experiment


def get_datasets():
    """Returns all the data sets."""
    dataset_name = h.fixed("dataset", "3dshapes")
    multithread = h.fixed("multithread", False)
    resize = h.fixed("resize", None)
    center_crop = h.fixed("center_crop", None)
    decoder_distribution = h.fixed("decoder_distribution", "cross-entropy")
    config_gaussian = h.zipit([dataset_name, decoder_distribution, multithread, resize, center_crop])

    return h.chainit([config_gaussian])



def get_num_latent(sweep):
    return h.sweep("latent_dim", h.discrete(sweep))



def get_seeds(num):
    """Returns random seeds."""
    return h.sweep("random_seed", h.categorical(list(range(num))))



def get_default_models():
    # BetaVAE config.
    model_name = h.fixed("method", "EFFICIENTWEAKVAE")
    filters = h.fixed("n_filters", 128)
    betas = h.sweep("beta", h.discrete([1., 2.]))

    warm_up = h.fixed("warm_up_iterations", 0)

    config_beta_vae = h.zipit([model_name, betas,filters, warm_up])

    return h.chainit([config_beta_vae])



def get_config():
    """Returns the hyperparameter configs for different experiments."""

    batch_size = h.fixed("batch_size", 64)
    lr = h.fixed("lr", 0.0001)

    wd = h.fixed("wd", 0.0)  # 1e-11
    #epochs = h.fixed("epochs", 26) # 26

    epochs = h.fixed("iterations", 50000) # 300000

    scheduler = h.fixed("scheduler_name", "reduceonplateau")
    loss = h.fixed("loss", "mse")
    factors_idx = h.fixed("factor_idx", list(range(0, 6)))  # consider all the factors
    k = h.fixed("k", 1)
    aggregator = h.sweep("aggregator", h.discrete(["labels"])) # "argmax",


    return h.product([
        get_datasets(),
        batch_size,
        lr,
        wd,
        epochs,
        loss,
        factors_idx,
        get_default_models(),
        get_num_latent([10]),
        get_seeds(10), # 5
        k,
        aggregator,
        scheduler
    ])


def get_config_postprocess():
    mode = h.fixed("mode", ["mean", "sampled"])
    num_representation = h.fixed("num_representation", 16)
    factors_idx = h.fixed("factor_idx_process", list(range(0, 6)))

    label_noise = h.fixed("label_noise", ["perfect"]) # , "bin", "noisy", "permuted"
    

    return h.product([
        factors_idx,
	label_noise,
        get_datasets(),
        get_default_models(),
        get_num_latent([10]),
        get_seeds(1),
        mode,
        num_representation
    ])[0]


def get_metrics():
    return h.fixed("metrics", ["ccd", "ccd_factors", "mig", "dci-disentanglement", "beta_vae", "factor_vae", "gbt_regressor", "mlp_regressor", "gbt_regressor_pruned", "mlp_regressor_pruned"])  #


def get_config_eval():
    mode = h.fixed("mode_eval", ["mean"])
    label_noise = h.fixed("label_noise", ["perfect"]) # , "bin", "noisy", "permuted"

    return h.product([
        label_noise,
        get_metrics(),
        get_seeds(1),
        mode
    ])[0]


class SHAPE3D(Experiment):

    def __init__(self):
        super(SHAPE3D, self).__init__()

    def get_number_sweep(self):
        return len(get_config())

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = get_config()[model_num]
        model = config

        return model

    def get_postprocess_config(self):
        """Returns postprocessing config files."""

        return get_config_postprocess()

    def get_eval_config(self):
        """Returns evaluation config files."""
        return get_config_eval()
