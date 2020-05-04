from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import torch

from spaces import Categoric
from schedulers import Scheduler
from schedulers import GridSearcher
from schedulers import BayesianOptimizer
from early_stoppers import ConstantStopper
from early_stoppers import StableStopper
from algorithms import GCNAlgo
from ensemblers import Ensembler
from utils import fix_seed, generate_pyg_data, divide_data

logger = logging.getLogger('code_submission')
logger.setLevel('DEBUG')
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s\t%(levelname)s %(filename)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

ALGO = GCNAlgo
STOPPER = StableStopper
SCHEDULER = BayesianOptimizer
ENSEMBLER = Ensembler
FRAC_FOR_SEARCH=0.75

# TO DO: empirically check the correctness of seeding
fix_seed(1234)


class Model(object):

    def __init__(self):
        """Constructor
        only `train_predict()` is measured for timing, put as much stuffs
        here as possible
        """

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')
        # so tricky...
        a_cpu = torch.ones((10,), dtype=torch.float32)
        a_gpu = a_cpu.to(self.device)

        self._hyperparam_space = ALGO.hyperparam_space
        # used by the scheduler for deciding when to stop each trial
        early_stopper = STOPPER(max_step=800)
        # ensemble the promising models searched
        ensembler = ENSEMBLER(
            config_selection='greedy', training_strategy='cv')
        # schedulers conduct HPO
        # current implementation: HPO for only one model
        self._scheduler = SCHEDULER(self._hyperparam_space, early_stopper, ensembler)

    def train_predict(self, data, time_budget, n_class, schema):
        """the only way ingestion interacts with user script"""

        self._scheduler.setup_timer(time_budget)
        data = generate_pyg_data(data, False, False).to(self.device)
        train_mask, early_valid_mask, final_valid_mask = divide_data(data, [7, 1, 2], self.device)
        logger.info("remaining {}s after data prepreration".format(self._scheduler.get_remaining_time()))

        algo = None
        while not self._scheduler.should_stop(FRAC_FOR_SEARCH):
            if algo:
                # within a trial, just continue the training
                train_info = algo.train(data, train_mask)
                early_stop_valid_info = algo.valid(data, early_valid_mask)
                if self._scheduler.should_stop_trial(train_info, early_stop_valid_info):
                    valid_info = algo.valid(data, final_valid_mask)
                    self._scheduler.record(algo, valid_info)
                    algo = None
            else:
                # trigger a new trial
                config = self._scheduler.get_next_config()
                if config:
                    algo = ALGO(n_class, data.x.size()[1], self.device, config)
                else:
                    # have exhausted the search space
                    break
        if algo is not None:
            valid_info = algo.valid(data, final_valid_mask)
            self._scheduler.record(algo, valid_info)
        logger.info("remaining {}s after HPO".format(self._scheduler.get_remaining_time()))
        
        pred = self._scheduler.pred(
            n_class, data.x.size()[1], self.device, data, ALGO, True)
        logger.info("remaining {}s after ensemble".format(self._scheduler.get_remaining_time()))

        return pred
