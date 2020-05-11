from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import copy

from spaces import Categoric, Numeric

logger = logging.getLogger('code_submission')


class Scheduler(object):
    """Base class for all schedulers
    Provides unified interfaces for our `Model`
    """

    def __init__(self,
                 hyperparam_space,
                 early_stopper,
                 ensembler):

        self._hyperparam_space = hyperparam_space
        self._early_stopper = early_stopper
        self._ensembler = ensembler
        self._results = list()

    def setup_timer(self, time_budget):
        self._start_time = time.time()
        self._time_budget = time_budget

    def get_remaining_time(self):
        cur = time.time()
        return self._time_budget - cur + self._start_time

    def should_stop(self, frac_for_search=0.85):
        """Judge whether the HPO procedure should be stopped
        Arguments:
            frac_for_search (float): the fraction of time budget for HPO
        """

        cur_time = time.time()
        return (cur_time - self._start_time) >= frac_for_search * self._time_budget

    def reset_trial(self):
        self._early_stopper.reset()

    def get_next_config(self):
        """Provide the config for instantiating a trial
        Each subclass could override this method and propose the config
        by some fancy HPO algorithm
        """

        self.reset_trial()
        self._cur_config = self.get_default()
        return self._cur_config if len(self._results) == 0 else None

    def should_stop_trial(self, train_info, early_stop_valid_info):
        should_early_stop = self._early_stopper.should_early_stop(
            train_info, early_stop_valid_info)
        return should_early_stop

    def record(self, algo, valid_info):
        """record (config, ckpt_path, valid_info, #epochs) for a trial"""

        path = "team_common_hpo_{}.pt".format(len(self._results))
        algo.save_model(path)
        self._results.append(
            (copy.deepcopy(self._cur_config), path, valid_info, self._early_stopper.get_cur_step()))

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

    def pred(self, n_class, num_features, device, data, algo, learn_from_scratch=False, non_hpo_config=dict()):
        considered_configs = self._ensembler.select_configs(self._results)
        predictions = self._ensembler.ensemble(
            n_class, num_features, device, data, self, algo,
            considered_configs, learn_from_scratch, non_hpo_config)
        return predictions
