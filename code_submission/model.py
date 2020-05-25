from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import torch

# from algorithms import GraphSAINTRandomWalkSampler
from algorithms import GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo, AdaGCNAlgo
from algorithms.model_selection import select_algo_from_data
from spaces import Categoric
from schedulers import *
from early_stoppers import *
from algorithms import GCNAlgo
from ensemblers import Ensembler
from utils import *
from torch_geometric.data import DataLoader
import subprocess
import os

logger = logging.getLogger('code_submission')
logger.setLevel('DEBUG')
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s\t%(levelname)s %(filename)s: %(message)s"))
logger.addHandler(handler)
logger.propagate = False

GCN_VERSIONs = ["dgl_gcn", "pyg_gcn"]
GCN_VERSION = GCN_VERSIONs[0]
ALGOs = [GCNAlgo, SplineGCNAlgo, SplineGCN_APPNPAlgo, AdaGCNAlgo]
ALGO = ALGOs[0]
STOPPERs = [MemoryStopper, NonImprovementStopper, StableStopper, EmpiricalStopper]
HPO_STOPPER = STOPPERs[0]
ENSEMBLER_STOPPER = STOPPERs[3]
SCHEDULERs = [GridSearcher, BayesianOptimizer, Scheduler, GeneticOptimizer]
SCHEDULER = SCHEDULERs[3]
ENSEMBLER = Ensembler
FEATURE_ENGINEERING = True
LEARN_FROM_SCRATCH = False
# todo (daoyuan) dynamic Frac_for_search, on dataset d, GCN has not completed even one entire training,
#  to try set more time budget fot those big graph.
FRAC_FOR_SEARCH = 0.75
FIX_FOCAL_LOSS = False
DATA_SPLIT_RATE = [7, 1, 2]
DATA_SPLIT_FOR_EACH_TRIAL = True
SAVE_TEST_RESULTS = True
CONSIDER_DIRECTED_GCN = False
CONDUCT_MODEL_SELECTION = False
LOG_BEST = True

# loader = GraphSAINTRandomWalkSampler(data, batch_size=1000, walk_length=5,
#                                      num_steps=5, sample_coverage=1000,
#                                      save_dir=,
#                                      num_workers=4)


class Model(object):

    def __init__(self, seed=time.time()):
        """Constructor
        only `train_predict()` is measured for timing, put as much stuffs
        here as possible
        """

        # convenient for comparing solutions
        logger.info("seeding with {}".format(seed))
        fix_seed(int(seed))

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
        # self.ensembler = ENSEMBLER(
        #     early_stopper=self.ensembler_early_stopper, config_selection='top10', training_strategy='naive')
        self.ensembler = ENSEMBLER(
            early_stopper=self.ensembler_early_stopper, config_selection='auto', training_strategy='hybrid', return_best=LOG_BEST)
        # schedulers conduct HPO
        # current implementation: HPO for only one model
        self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
        self.non_hpo_config = {'LEARN_FROM_SCRATCH': LEARN_FROM_SCRATCH,
                               "gcn_version": GCN_VERSION}

        try:
            self.cp_cnpy_file()
        except Exception as err_msg:
            logger.info('copy files failed with error msg: {}'.format(err_msg))

    def cp_cnpy_file(self):
        file_path = os.path.join(os.path.dirname(__file__),'cnpy_file')
        file_name = ['libcnpy.so', 'libcnpy.a', 'cnpy.h', 'mat2npz', 'npy2mat', 'npz2mat']
        file_name = [os.path.join(file_path, each_file_name) for each_file_name in file_name]
        
        run_commands = ' '.join(['cp', file_name[0], file_name[1], '/usr/local/lib/'])
        cmd_return = subprocess.run(run_commands, shell=True)

        run_commands = ' '.join(['cp', file_name[2], '/usr/local/include/'])
        cmd_return = subprocess.run(run_commands, shell=True)

        run_commands = ' '.join(['cp', file_name[3], file_name[4], file_name[5], '/usr/local/bin/'])
        cmd_return = subprocess.run(run_commands, shell=True)
        
        os.environ['LD_LIBRARY_PATH'] = '%s:%s'%('$LD_LIBRARY_PATH','/usr/local/lib')

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
        self.imbalanced_task_type, self.is_minority_class = get_imbalanced_task_type(train_y, n_class)

        if FEATURE_ENGINEERING:
            data = generate_pyg_data(data, n_class, time_budget).to(self.device)
        else:
            data = generate_pyg_data_without_transform(data).to(self.device)

        self.non_hpo_config["label_alpha"] = label_weights
        # is_undirected = data.is_undirected()
        # data["directed"] = not is_undirected  # used for directed DGL-GCN

        is_real_weighted_graph = not (int(torch.sum(data.edge_weight)) == data.edge_index.shape[1])
        data["real_weight_edge"] = is_real_weighted_graph

        self.non_hpo_config["directed"] = not is_undirected and CONSIDER_DIRECTED_GCN
        logger.info("The graph is {}directed graph".format("un-" if is_undirected else ""))
        logger.info("The graph is {} weighted edge graph".format("real" if is_real_weighted_graph else "fake"))
        logger.info("The graph has {} nodes and {} edges".format(data.num_nodes, data.edge_index.size(1)))
        logger.info("Your gcn_version is {}".format(GCN_VERSION))

        global ALGO
        if CONDUCT_MODEL_SELECTION:
            suiable_algo, suitable_non_hpo_config = select_algo_from_data(ALGOs, data, self.non_hpo_config)
            self.non_hpo_config = suitable_non_hpo_config
            if suiable_algo != ALGO:
                remain_time_budget = self._scheduler.get_remaining_time()
                self.change_algo(suiable_algo, remain_time_budget)
                ALGO = suiable_algo
        # loader = DataLoader(data, batch_size=32, shuffle=True)

        change_hyper_space = ALGO.ensure_memory_safe(data.x.size()[0], data.edge_weight.size()[0],
                                                     n_class, data.x.size()[1], not is_undirected)
        if change_hyper_space:
            self._hyperparam_space = ALGO.hyperparam_space
            logger.info('Changed algo hyperparam_space: %s', hyperparam_space_tostr(ALGO.hyperparam_space))
            remain_time_budget = self._scheduler.get_remaining_time()
            self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
            self._scheduler.setup_timer(remain_time_budget)

        global FRAC_FOR_SEARCH
        global DATA_SPLIT_RATE
        global LOG_BEST
        if self.imbalanced_task_type == 1:
            FRAC_FOR_SEARCH = 0.95
            DATA_SPLIT_RATE = [1, 0, 0]
            LOG_BEST = True
            ALGO.hyperparam_space['loss_type'] = Categoric(["focal_loss"], None, "focal_loss")
            ALGO.hyperparam_space['res_type'].default_value = 1.0
            self._hyperparam_space = ALGO.hyperparam_space
            self.hpo_early_stopper = AdaptiveWeightStopper()
            self.ensembler = ENSEMBLER(
                early_stopper=self.ensembler_early_stopper, config_selection='auto',
                training_strategy='hpo_trials', return_best=LOG_BEST)
            remain_time_budget = self._scheduler.get_remaining_time()
            self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
            self._scheduler.setup_timer(remain_time_budget)
            self.non_hpo_config['is_minority'] = self.is_minority_class
        elif self.imbalanced_task_type == 2:
            DATA_SPLIT_RATE = [7, 1, 2]
            LOG_BEST = False
            ALGO.hyperparam_space['loss_type'] = Categoric(["focal_loss"], None, "focal_loss")
            ALGO.hyperparam_space['res_type'] = Categoric([0., 1.], None, 0.)
            ALGO.hyperparam_space['num_layers'] = Categoric(list(range(1, 4)), None, 2)
            ALGO.hyperparam_space['wide_and_deep'] = Categoric(['deep'], None, "deep")
            self._hyperparam_space = ALGO.hyperparam_space
            remain_time_budget = self._scheduler.get_remaining_time()
            self._scheduler = SCHEDULER(self._hyperparam_space, self.hpo_early_stopper, self.ensembler)
            self._scheduler.setup_timer(remain_time_budget)

        train_mask, early_valid_mask, final_valid_mask = None, None, None
        if not DATA_SPLIT_FOR_EACH_TRIAL:
            train_mask, early_valid_mask, final_valid_mask = divide_data_label_wise(
                data, DATA_SPLIT_RATE, self.device, n_class, train_y)
            if DATA_SPLIT_RATE[1] == 0.0 and DATA_SPLIT_RATE[2] == 0.0:
                early_valid_mask = train_mask
                final_valid_mask = train_mask
            elif DATA_SPLIT_RATE[1] == 0.0 and DATA_SPLIT_RATE[2] != 0.0:
                early_valid_mask = final_valid_mask
            elif DATA_SPLIT_RATE[1] != 0.0 and DATA_SPLIT_RATE[2] == 0.0:
                final_valid_mask = early_valid_mask

        logger.info("remaining {}s after data preparation".format(self._scheduler.get_remaining_time()))
        logger.info('Device: %s', self.device)
        logger.info('FRAC_FOR_SEARCH: %s', FRAC_FOR_SEARCH)
        logger.info('Feature engineering: %s', FEATURE_ENGINEERING)
        logger.info('Fix focal loss: %s', FIX_FOCAL_LOSS)
        logger.info('Default Algo is: %s', ALGO)
        logger.info('Algo hyperparam_space: %s', hyperparam_space_tostr(ALGO.hyperparam_space))
        logger.info('HPO_Early_stopper: %s', type(self.hpo_early_stopper).__name__)
        logger.info('Ensembler_Early_stopper: %s', type(self.ensembler_early_stopper).__name__)
        logger.info('Ensembler: %s', type(self.ensembler).__name__)
        logger.info('Learn from scratch in ensembler: %s', self.non_hpo_config["LEARN_FROM_SCRATCH"])

        algo = None
        tmp_results = None
        tmp_valid_info = None
        while not self._scheduler.should_stop(FRAC_FOR_SEARCH):
            if algo:
                # within a trial, just continue the training
                T = self._scheduler._early_stopper.get_T() if self.imbalanced_task_type == 1 else 1.0
                train_info = algo.train(data, train_mask, T)
                early_stop_valid_info = algo.valid(data, early_valid_mask)
                if LOG_BEST and self._scheduler._early_stopper.should_log(train_info, early_stop_valid_info):
                    tmp_results = algo.pred(data, make_decision=False)
                    tmp_valid_info = algo.valid(data, final_valid_mask) if DATA_SPLIT_RATE[2] != 0.0 else early_stop_valid_info
                if self._scheduler.should_stop_trial(train_info, early_stop_valid_info):
                    # valid_info = algo.valid(data, final_valid_mask) if not LOG_BEST else tmp_valid_info
                    valid_info = algo.valid(data, final_valid_mask)
                    test_results = None
                    if SAVE_TEST_RESULTS:
                        test_results = algo.pred(data, make_decision=False) if not LOG_BEST else tmp_results
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
                        train_mask, early_valid_mask, final_valid_mask = divide_data_label_wise(
                            data, DATA_SPLIT_RATE, self.device, n_class, train_y)
                        if DATA_SPLIT_RATE[1] == 0.0 and DATA_SPLIT_RATE[2] == 0.0:
                            early_valid_mask = train_mask
                            final_valid_mask = train_mask
                        elif DATA_SPLIT_RATE[1] == 0.0 and DATA_SPLIT_RATE[2] != 0.0:
                            early_valid_mask = final_valid_mask
                        elif DATA_SPLIT_RATE[1] != 0.0 and DATA_SPLIT_RATE[2] == 0.0:
                            final_valid_mask = early_valid_mask
                else:
                    # have exhausted the search space
                    break
        if algo is not None:
            # valid_info = algo.valid(data, final_valid_mask) if not LOG_BEST else tmp_valid_info
            valid_info = algo.valid(data, final_valid_mask)
            test_results = None
            if SAVE_TEST_RESULTS:
                test_results = algo.pred(data, make_decision=False) if not LOG_BEST else tmp_results
            self._scheduler.record(algo, valid_info, test_results)

        logger.info("remaining {}s after HPO".format(self._scheduler.get_remaining_time()))

        pred = self._scheduler.pred(
            n_class, data.x.size()[1], self.device, data, ALGO,
            self.non_hpo_config["LEARN_FROM_SCRATCH"], self.non_hpo_config,
            train_y)
        logger.info("remaining {}s after ensemble".format(self._scheduler.get_remaining_time()))

        return pred
