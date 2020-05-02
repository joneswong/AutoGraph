from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch

from schedulers import Scheduler, GridSearcher, BayesianOptimizer
from early_stoppers import ConstantStopper
from early_stoppers import StableStopper
# from algorithms import GraphSAINTRandomWalkSampler
from algorithms import GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo
from ensemblers import GreedyStrategy
from utils import fix_seed, generate_pyg_data, generate_pyg_data_feature_transform, \
    divide_data, hyperparam_space_tostr


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s\t%(levelname)s %(filename)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

ALGOs = [GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo]
ALGO = ALGOs[1]
STOPPERs = [StableStopper, ConstantStopper]
STOPPER = STOPPERs[1]
SCHEDULERs = [GridSearcher, BayesianOptimizer, Scheduler]
SCHEDULER = SCHEDULERs[2]
ENSEMBLER = GreedyStrategy
FEATURE_ENGINEERING = False
FRAC_FOR_SEARCH = 0.85

# loader = GraphSAINTRandomWalkSampler(data, batch_size=1000, walk_length=5,
#                                      num_steps=5, sample_coverage=1000,
#                                      save_dir=,
#                                      num_workers=4)

fix_seed(1234)

class Model(object):

    def __init__(self):
        """Constructor
        only `train_predict()` is measured for timing, put as much stuffs
        here as possible
        """

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')
        self._hyperparam_space = ALGO.hyperparam_space

        # used by the scheduler for deciding when to stop each trial
        early_stopper = STOPPER(max_step=800)

        # schedulers conduct HPO
        # current implementation: HPO for only one model
        self._scheduler = SCHEDULER(self._hyperparam_space, early_stopper)

        # ensemble the promising models searched
        self._ensembler = ENSEMBLER(finetune=False)

        logger.info('FRAC_FOR_SEARCH: %s', FRAC_FOR_SEARCH)
        logger.info('Feature engineering: %s', FEATURE_ENGINEERING)
        logger.info('Algo hyperparam_space: %s', hyperparam_space_tostr(ALGO.hyperparam_space))
        logger.info('Scheduler: %s', type(self._scheduler).__name__)
        logger.info('Ensembler: %s', type(self._ensembler).__name__)

    def train_predict(self, data, time_budget, n_class, schema):
        """the only way ingestion interacts with user script"""

        self._scheduler.setup_timer(time_budget)
        if FEATURE_ENGINEERING:
            data = generate_pyg_data_feature_transform(data).to(self.device)
        else:
            data = generate_pyg_data(data).to(self.device)
        train_mask, early_valid_mask, final_valid_mask = divide_data(data, [7,1,2])
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
        
        final_algo = self._ensembler.boost(
            n_class, data.x.size()[1], self.device,
            data, self._scheduler.get_results(), ALGO)
        logger.info("remaining {}s after ensemble".format(self._scheduler.get_remaining_time()))

        pred = final_algo.pred(data)

        return pred.cpu().numpy().flatten()

