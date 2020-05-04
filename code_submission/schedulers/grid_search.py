from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from schedulers import Scheduler
from spaces import Categoric, Numeric


class GridSearcher(Scheduler):
    """Apply grid search for HPO"""

    def __init__(self,
                 hyperparam_space,
                 early_stopper,
                 ensembler):

        super(GridSearcher, self).__init__(
            hyperparam_space, early_stopper, ensembler)

        # grid search for categorical parameters while the numeric parameters are set as default value
        self.discrete_keys = []
        for key, value in self._hyperparam_space.items():
            if isinstance(value, Categoric) and len(value.categories) != 0:
                self.discrete_keys.append(key)

        self.total_discrete_param = np.product([len(self._hyperparam_space[key].categories)
                                                for key in self.discrete_keys])
        self.discrete_param_count = 0
        self._cur_config = self.get_default()

    def get_next_config(self):
        self._early_stopper.reset()

        param_index = self.discrete_param_count
        for key in self.discrete_keys:
            values = self._hyperparam_space[key].categories
            self._cur_config[key] = values[param_index % len(values)]
            param_index = int(param_index / len(values))
        self.discrete_param_count += 1

        return self._cur_config if self.discrete_param_count <= self.total_discrete_param else None
