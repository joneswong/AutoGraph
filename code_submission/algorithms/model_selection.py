from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo


LARGE_EDGE_NUMBER = 500000

ALGOs = [GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo]


def select_algo_from_data(ALGOs, data):
    """
    Heuristic rules to select algo to be searched, based on the data property

    :param ALGOs: candidate algos
    :param data:
    :return: suitable algo
    """

    if data.edge_index.size(1) > LARGE_EDGE_NUMBER:
        return GCNAlgo
    else:
        return SplineGCNAlgo

    # according to the steps/layers to be propagated/searched, we may need APPNP algo
