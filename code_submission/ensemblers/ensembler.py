from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch

from utils import get_performance, divide_data

logger = logging.getLogger('code_submission')

CV_NUM_FOLD=5
SAFE_FRAC=0.95
SMALL_EPOCH=20
FINE_TUNE_WHEN_CV=True


class Ensembler(object):

    def __init__(self,
                 early_stopper,
                 config_selection='greedy',
                 training_strategy='cv',
                 *args,
                 **kwargs):

        self._ensembler_early_stopper = early_stopper
        self._config_selection = config_selection
        self._training_strategy = training_strategy

    def select_configs(self, results):
        """Select configs for training the final model(s)
            Arguments:
                results (list): each element is a tuple of config (Space), \
                                path (str), performance (dict).
            Returns: a list of the input element(s)
        """

        logger.info("to select config(s) from {} candidates".format(len(results)))
        if self._config_selection == 'greedy':
            sorted_results = sorted(results, key=lambda x: get_performance(x[2]))
            optimal = sorted_results[-1]
            return [optimal]
        else:
            # TO DO: provide other strategies
            pass

    def ensemble(self,
                 n_class,
                 num_features,
                 device,
                 data,
                 scheduler,
                 algo,
                 opt_records,
                 learn_from_scratch=False,
                 non_hpo_config=dict()
                 ):

        logger.info("to train model(s) with {} config(s)".format(len(opt_records)))
        for opt_record in opt_records:
            logger.info("searched opt_config is {}.".format(opt_records))
        if self._training_strategy == 'cv':
            opt_record = opt_records[0]
            parts = divide_data(data, CV_NUM_FOLD*[10/CV_NUM_FOLD], device)
            part_logits = list()
            cur_valid_part_idx = 0
            while (not scheduler.should_stop(SAFE_FRAC)) and (cur_valid_part_idx < CV_NUM_FOLD):
                model = algo(n_class, num_features, device, opt_record[0], non_hpo_config)
                if not learn_from_scratch:
                    model.load_model(opt_record[1])
                train_mask = torch.sum(
                    torch.stack([m for i, m in enumerate(parts) if i != cur_valid_part_idx]), 0).type(torch.bool)
                valid_mask = parts[cur_valid_part_idx]
                self._ensembler_early_stopper.reset()
                while not scheduler.should_stop(SAFE_FRAC):
                    train_info = model.train(data, train_mask)
                    valid_info = model.valid(data, valid_mask)
                    if self._ensembler_early_stopper.should_early_stop(train_info, valid_info):
                        logits = model.pred(data, make_decision=False)
                        part_logits.append(logits.cpu().numpy())
                        break
                if FINE_TUNE_WHEN_CV:
                    # naive version: enhance the model by train with the valid data part
                    while not scheduler.should_stop(SAFE_FRAC):
                        # todo: add some other heuristic method to set the small_epoch,
                        #  e.g., the average epochs of the stopper
                        for i in range(SMALL_EPOCH):
                            model.train(data, valid_mask)
                cur_valid_part_idx += 1
            if len(part_logits) == 0:
                logger.warn("have not completed even one training course")
                logits = model.pred(data, make_decision=False)
                part_logits.append(logits.cpu().numpy())
            logger.info("ensemble {} models".format(len(part_logits)))
            pred = np.argmax(np.mean(np.stack(part_logits), 0), -1).flatten()
            return pred
        elif self._training_strategy == 'naive':
            # just train a model with the optimal config on the whole labeled samples
            opt_record = opt_records[0]
            model = algo(n_class, num_features, device, opt_record[0])
            if not learn_from_scratch:
                model.load_model(opt_record[1])
            self._ensembler_early_stopper.reset()
            while not scheduler.should_stop(SAFE_FRAC):
                train_info = model.train(data, data.train_mask)
                # currently, this only cooperates with fixed #epochs
                if self._ensembler_early_stopper.should_early_stop(train_info, None) and \
                   self._ensembler_early_stopper.get_cur_step() >= opt_record[3]:
                    logpr = model.pred(data, make_decision=False)
                    break
            logger.info("the final model traverses the whole training data for {} epochs".format(self._ensembler_early_stopper.get_cur_step()))
            pred = torch.argmax(logpr, -1).cpu().numpy().flatten()
            return pred
        else:
            # TO DO: provide other strategies
            pass
