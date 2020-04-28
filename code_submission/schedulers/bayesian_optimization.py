from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from schedulers import Scheduler


class BayesianOptimizer(Scheduler):
    """Apply Bayesian optimization for HPO"""
    # (jones.wz) TO DO: implementation

    def __init__(self,
                 hyperparam_space,
                 default_config):

        super(BayesianOptimizer, self).__init__(
            hyperparam_space, default_config)
