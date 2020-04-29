from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy

from spaces import Categoric, Numeric


class Scheduler(object):
    """Base class for all schedulers
    Provides unified interfaces for our `Model`
    """

    def __init__(self,
                 hyperparam_space,
                 early_stopper):

        self._hyperparam_space = hyperparam_space
        self._early_stopper = early_stopper
        self._results = list()

    def setup_timer(self, time_budget):
        self._start_time = time.time()
        self._time_budget = time_budget

    def should_stop(self, frac_for_search=0.85):
        """Judge whether the HPO procedure should be stopped
        Arguments:
            frac_for_search (float): the fraction of time budget for HPO
        """

        cur_time = time.time()
        return (cur_time - self._start_time) >= frac_for_search * self._time_budget

    def get_next_config(self):
        """Provide the config for instantiating a trial
        Each subclass could override this method and propose the config
        by some fancy HPO algorithm
        """

        self._early_stopper.reset()
        self._cur_config = self.get_default()
        return self._cur_config if len(self._results) == 0 else None

    def should_stop_trial(self, train_info, early_stop_valid_info):
        should_early_stop = self._early_stopper.should_early_stop(
            train_info, early_stop_valid_info)
        return should_early_stop

    def record(self, algo, valid_info):
        path = "team_common_hpo_{}.pt".format(len(self._results))
        algo.save_model(path)
        self._results.append((copy.deepcopy(self._cur_config), path, valid_info))

    def get_results(self):
        return self._results

    def get_default(self):
        results = dict()
        for k, v in self._hyperparam_space.items():
            if isinstance(v, Numeric):
                results[k] = v.default_value
            else:
                if v.subspaces:
                    results[k] = (v.default_value, self.get_default(v.subspaces[v.categories.index(v.default_value)]))
                else:
                    results[k] = v.default_value
        return results
