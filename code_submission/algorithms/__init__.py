from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from algorithms.gcn_algo import GCNAlgo


agents = dict(
    GCNAlgo=GCNAlgo,
    )

__all__ = ["GCNAlgo"]
