from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import torch


# from algorithms import GraphSAINTRandomWalkSampler
from algorithms import GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo
from algorithms.model_selection import select_algo_from_data
from spaces import Categoric
from schedulers import *
from early_stoppers import *
from algorithms import GCNAlgo
from ensemblers import Ensembler
from utils import *
from torch_geometric.data import DataLoader


logger = logging.getLogger('code_submission')
logger.setLevel('DEBUG')
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s\t%(levelname)s %(filename)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

ALGOs = [GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo]
ALGO = ALGOs[0]
STOPPERs = [MemoryStopper, NonImprovementStopper, StableStopper, EmpiricalStopper]
HPO_STOPPER = STOPPERs[3]
ENSEMBLER_STOPPER = STOPPERs[3]
SCHEDULERs = [GridSearcher, BayesianOptimizer, Scheduler, GeneticOptimizer]
SCHEDULER = SCHEDULERs[3]
ENSEMBLER = Ensembler
FEATURE_ENGINEERING = False
non_hpo_config = dict()
non_hpo_config["LEARN_FROM_SCRATCH"] = True
# todo (daoyuan) dynamic Frac_for_search, on dataset d, GCN has not completed even one entire training,
#  to try set more time budget fot those big graph.
FRAC_FOR_SEARCH = 0.75
FIX_FOCAL_LOSS = False
DATA_SPLIT_FOR_EACH_TRIAL = True
SAVE_TEST_RESULTS = True
TOP_K = 10

# loader = GraphSAINTRandomWalkSampler(data, batch_size=1000, walk_length=5,
#                                      num_steps=5, sample_coverage=1000,
#                                      save_dir=,
#                                      num_workers=4)

# fix_seed(1234)


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
        self.hpo_early_stopper = HPO_STOPPER(max_step=500)
        self.ensembler_early_stopper = ENSEMBLER_STOPPER()
        # ensemble the promising models searched
        # self.ensembler = ENSEMBLER(
        #     early_stopper=self.ensembler_early_stopper, config_selection='greedy', training_strategy='cv')
        self.ensembler = ENSEMBLER(
            early_stopper=self.ensembler_early_stopper, config_selection='top_k', training_strategy='hpo_trials', top_k=TOP_K)
        # schedulers conduct HPO
        # current implementation: HPO for only one model
        self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
        self.non_hpo_config = non_hpo_config

        logger.info('Device: %s', self.device)
        logger.info('FRAC_FOR_SEARCH: %s', FRAC_FOR_SEARCH)
        logger.info('Feature engineering: %s', FEATURE_ENGINEERING)
        logger.info('Fix focal loss: %s', FIX_FOCAL_LOSS)
        logger.info('Default Algo is: %s', ALGO)
        logger.info('Algo hyperparam_space: %s', hyperparam_space_tostr(ALGO.hyperparam_space))
        logger.info('HPO_Early_stopper: %s', type(self.hpo_early_stopper).__name__)
        logger.info('Ensembler_Early_stopper: %s', type(self.ensembler_early_stopper).__name__)
        logger.info('Ensembler: %s', type(self.ensembler).__name__)
        logger.info('Learn from scratch in ensembler: %s', non_hpo_config["LEARN_FROM_SCRATCH"])

    def change_algo(self, ALGO, remain_time_budget):
        self._hyperparam_space = ALGO.hyperparam_space
        logger.info('Change to algo: %s', ALGO)
        logger.info('Changed algo hyperparam_space: %s', hyperparam_space_tostr(ALGO.hyperparam_space))
        self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
        self._scheduler.setup_timer(remain_time_budget)

    def train_predict(self, data, time_budget, n_class, schema):
        """the only way ingestion interacts with user script"""

        self._scheduler.setup_timer(time_budget)

        train_y = data['train_label'][['label']].to_numpy()
        label_weights = get_label_weights(train_y, n_class)

        if FEATURE_ENGINEERING:
            data = generate_pyg_data(data, n_class).to(self.device)
        else:
            data = generate_pyg_data_without_transform(data).to(self.device)

        train_mask, early_valid_mask, final_valid_mask = None, None, None
        if not DATA_SPLIT_FOR_EACH_TRIAL:
            # train_mask, early_valid_mask, final_valid_mask = divide_data(data, [7, 1, 2], self.device)
            train_mask, early_valid_mask, final_valid_mask = divide_data_label_wise(
                data, [7, 1, 2], self.device, n_class, train_y)
        logger.info("remaining {}s after data preparation".format(self._scheduler.get_remaining_time()))

        self.non_hpo_config["label_alpha"] = label_weights
        logger.info("The graph is {}directed graph".format("un-" if data.is_undirected() else ""))
        logger.info("The graph has {} nodes and {} edges".format(data.num_nodes, data.edge_index.size(1)))
        suiable_algo, suitable_non_hpo_config = select_algo_from_data(ALGOs, data, self.non_hpo_config)
        self.non_hpo_config = suitable_non_hpo_config
        global ALGO
        if suiable_algo != ALGO:
            remain_time_budget = self._scheduler.get_remaining_time()
            self.change_algo(suiable_algo, remain_time_budget)
            ALGO = suiable_algo
        # loader = DataLoader(data, batch_size=32, shuffle=True)

        algo = None
        while not self._scheduler.should_stop(FRAC_FOR_SEARCH):
            if algo:
                # within a trial, just continue the training
                train_info = algo.train(data, train_mask)
                early_stop_valid_info = algo.valid(data, early_valid_mask)
                if self._scheduler.should_stop_trial(train_info, early_stop_valid_info):
                    valid_info = algo.valid(data, final_valid_mask)
                    test_results = None
                    if SAVE_TEST_RESULTS:
                        test_results = algo.pred(data, make_decision=False)
                    self._scheduler.record(algo, valid_info, test_results)
                    algo = None
            else:
                # trigger a new trial
                config = self._scheduler.get_next_config()

                if config:
                    if FIX_FOCAL_LOSS:
                        self.non_hpo_config["label_alpha"] = label_weights
                        config["loss_type"] = "focal_loss"
                    algo = ALGO(n_class, data.x.size()[1], self.device, config, self.non_hpo_config)
                    if DATA_SPLIT_FOR_EACH_TRIAL:
                        # train_mask, early_valid_mask, final_valid_mask = divide_data(data, [7, 1, 2], self.device)
                        train_mask, early_valid_mask, final_valid_mask = divide_data_label_wise(
                            data, [7, 1, 2], self.device, n_class, train_y)
                else:
                    # have exhausted the search space
                    break
        if algo is not None:
            valid_info = algo.valid(data, final_valid_mask)
            test_results = None
            if SAVE_TEST_RESULTS:
                test_results = algo.pred(data, make_decision=False)
            self._scheduler.record(algo, valid_info, test_results)
        logger.info("remaining {}s after HPO".format(self._scheduler.get_remaining_time()))

        pred = self._scheduler.pred(
            n_class, data.x.size()[1], self.device, data, ALGO, self.non_hpo_config["LEARN_FROM_SCRATCH"], self.non_hpo_config)
        logger.info("remaining {}s after ensemble".format(self._scheduler.get_remaining_time()))

        return pred
