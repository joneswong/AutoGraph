from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from schedulers import Scheduler

class GridSearcher(Scheduler):
    """Apply grid search for HPO"""
    # (jones.wz) TO DO: implementation

    def __init__(self,
                 hyperparam_space,
                 default_config=None):

        super(GridSearcher, self).__init__(
            hyperparam_space, default_config)
