from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from schedulers import Scheduler
from spaces import Categoric, Numeric
from utils import get_performance


class BayesianOptimizer(Scheduler):
    """Apply Bayesian optimization for HPO"""

    def __init__(self,
                 hyperparam_space,
                 early_stopper):

        super(BayesianOptimizer, self).__init__(
            hyperparam_space, early_stopper)

        pbounds = {}
        for key, value in self._hyperparam_space.items():
            if isinstance(value, Categoric):
                sorted_categories = sorted(value.categories)
                pbounds[key] = [0, len(sorted_categories)-1e-10]
                self._hyperparam_space[key].categories = sorted_categories
            elif isinstance(value, Numeric):
                assert value.low <= value.high
                pbounds[key] = [value.low, value.high]
            else:
                raise NotImplementedError

        self.bayesian_optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.bo_params = None
        self._cur_config = self.get_default()

    def get_next_config(self):
        self._early_stopper.reset()

        if len(self._results) != 0:
            self.bayesian_optimizer.register(params=self.bo_params, target=get_performance(self._results[-1][2]))
        self.bo_params = self.bayesian_optimizer.suggest(self.utility)
        self._cur_config.update(self.bo_params)
        for key, value in self._hyperparam_space.items():
            if isinstance(value, Categoric):
                self._cur_config[key] = self._hyperparam_space[key].categories[int(self.bo_params[key])]

        return self._cur_config
