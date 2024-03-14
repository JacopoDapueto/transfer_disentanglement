from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from src.evaluation.metrics.omes import OMES, OMESFactors
from src.evaluation.metrics.mig import MIG, MIGFactors
from src.evaluation.metrics.dci import DCI_disentanglement
from src.evaluation.metrics.downstream_task import GBT_regressor, MLP_regressor, GBT_regressor_pruned, MLP_regressor_pruned
from src.evaluation.metrics.beta_vae_score import BetaVaeScore
from src.evaluation.metrics.factor_vae_score import FactorVaeScore



def get_named_metric(name):

    if name == "omes":
        return OMES
    elif name == "omes_factors":
        return OMESFactors
    elif name == "mig":
        return MIG
    elif name == "beta_vae":
        return BetaVaeScore
    elif name == "factor_vae":
        return FactorVaeScore
    elif name == "mig_factors":
        return MIGFactors
    elif name == "dci-disentanglement":
        return DCI_disentanglement
    elif name == "gbt_regressor":
        return GBT_regressor
    elif name == "mlp_regressor":
        return MLP_regressor
    elif name == "gbt_regressor_pruned":
        return GBT_regressor_pruned
    elif name == "mlp_regressor_pruned":
        return MLP_regressor_pruned

    else:
        raise ValueError("Invalid metric name.")