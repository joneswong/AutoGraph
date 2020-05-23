from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from algorithms.gcn_algo import GCNAlgo
from algorithms.gnn_algo import SplineGCNAlgo, SplineGCN_APPNPAlgo, AdaGCNAlgo
# from algorithms.graph_saint_sampler import GraphSAINTRandomWalkSampler


agents = dict(
    GCNAlgo=GCNAlgo,
    SplineAlgo=SplineGCNAlgo,
    SplineGCN_APPNPAlgo=SplineGCN_APPNPAlgo,
    # GraphSAINTRandomWalkSampler=GraphSAINTRandomWalkSampler,
    )

# __all__ = ["GCNAlgo", "SplineGCNAlgo", "SplineGCN_APPNPAlgo", "GraphSAINTRandomWalkSampler"]
__all__ = ["GCNAlgo", "SplineGCNAlgo", "SplineGCN_APPNPAlgo", "AdaGCNAlgo"]
