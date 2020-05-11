from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo


LARGE_EDGE_NUMBER = 20000

ALGOs = [GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo]


def select_algo_from_data(ALGOs, data, non_hpo_config):
    """
    Heuristic rules to select algo to be searched, based on the data property

    :param ALGOs: candidate algos
    :param data:
    :param non_hpo_config: fixed hyper-parameters
    :return: suitable algo
    """
    if data.edge_index.size(1) > LARGE_EDGE_NUMBER * 10:
        non_hpo_config["LEARN_FROM_SCRATCH"] = False  # to save training time, utilize the checkpoint
    if data.edge_index.size(1) > LARGE_EDGE_NUMBER:
        return GCNAlgo, non_hpo_config
    else:
        return SplineGCNAlgo, non_hpo_config

    # according to the steps/layers to be propagated/searched, we may need APPNP algo
