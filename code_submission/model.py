from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from spaces import Categoric
from schedulers import Scheduler
from early_stoppers import ConstantStopper
from algorithms import GCNAlgo
from ensemblers import GreedyStrategy
from utils import fix_seed, generate_pyg_data

FRAC_FOR_SEARCH=0.85
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
        self._hyperparam_space = dict(
            algo=Categoric([GCNAlgo], [GCNAlgo.hyperparam_space], GCNAlgo))
        # used by the scheduler for deciding when to stop each trial
        early_stopper = ConstantStopper(max_step=800)
        # schedulers conduct HPO
        self._scheduler = Scheduler(self._hyperparam_space, early_stopper)
        # ensemble the promising models searched
        self._ensembler = GreedyStrategy(finetune=False)

    def train_predict(self, data, time_budget, n_class, schema):
        """the only way ingestion interacts with user script"""

        self._scheduler.setup_timer(time_budget)
        # TO DO: apply feature engineering to the data in a pluggable way
        data = generate_pyg_data(data).to(self.device)
        # TO DO: implement some method/class for splitting the data
        train_data, early_stop_valid_data, valid_data = data, None, data

        algo = None
        while not self._scheduler.should_stop(FRAC_FOR_SEARCH):
            if algo:
                # within a trial, just continue the training
                train_info = algo.train(train_data)
                if early_stop_valid_data:
                    early_stop_valid_info = algo.valid(early_stop_valid_data)
                else:
                    early_stop_valid_info = None
                if self._scheduler.should_stop_trial(train_info, early_stop_valid_info):
                    valid_info = algo.valid(valid_data)
                    self._scheduler.record(algo, valid_info)
                    algo = None
            else:
                # trigger a new trial
                config = self._scheduler.get_next_config()
                if config:
                    algo = config["algo"][0](
                        n_class, data.x.size()[1], self.device, config["algo"][1])
                else:
                    # have exhausted the search space
                    break
        if algo is not None:
            valid_info = algo.valid(valid_data)
            self._scheduler.record(algo, valid_info)
        
        final_algo = self._ensembler.boost(
            n_class, data.x.size()[1], self.device,
            data, self._scheduler.get_results())

        pred = final_algo.pred(data)

        return pred.cpu().numpy().flatten()
