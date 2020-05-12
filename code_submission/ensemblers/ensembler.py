from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn.functional as F

from early_stoppers import ConstantStopper
from utils import divide_data, calculate_config_dist

logger = logging.getLogger('code_submission')

CV_NUM_FOLD=5
SAFE_FRAC=0.95
FINE_TUNE_EPOCH=50
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

        if self._config_selection == 'greedy' and self._training_strategy == 'naive':
            assert isinstance(self._ensembler_early_stopper, ConstantStopper), "required to use ConstantStopper"

    def select_configs(self, results):
        """Select configs for training the final model(s)
            Arguments:
                results (list): each element is a tuple of config (Space), \
                                path (str), performance (dict).
            Returns: a list of the input element(s)
        """

        logger.info("to select config(s) from {} candidates".format(len(results)))
        sorted_results = sorted(results, key=lambda x: x[2])

        if self._config_selection == 'greedy':
            # choose the best one
            optimal = sorted_results[-1]
            return [optimal]
        elif self._config_selection.startswith("top"):
            if self._config_selection.endswith("lo"):
                # choose the topK local optimals where we approximate this by
                # sequentially select the most different config (w.r.t. the
                # previous one) from the top-3K candidates
                K = int(self._config_selection[3:-2])
                num_cands = min(len(sorted_results), 3*K)
                num_needed = min(len(sorted_results), K)
                picked = [sorted_results[-1]]
                last_selected = -1
                flags = (num_cands-1) * [1] + [0]
                while len(picked) < num_needed:
                    picked_idx = -1
                    cur_max_dist = -float('inf')
                    for i in range(-1, -(num_cands+1), -1):
                        if flags[i]:
                            dist = calculate_config_dist(
                                sorted_results[last_selected],
                                sorted_results[last_selected])
                            if dist > cur_max_dist:
                                cur_max_dist = dist
                                picked_idx = i
                    if picked_idx != -1:
                        picked.append(sorted_results[picked_idx])
                        last_selected = picked_idx
                        flags[picked_idx] = 0
                logger.info("choosed {}".format(','.join([str(v) for v in flags])))
                return picked
            else:
                # choose the topK
                K = min(len(sorted_results), int(self._config_selection[3:]))
                considered = sorted_results[-K:]
                considered.reverse() 
                return considered
        else:
            # provide other strategies
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
        """Training the model(s) for prediction
            Arguments:
                n_class (int): the number of categories
                num_features (int): the dimension of node representation
                device (obj): cpu or gpu
                data (obj): the graph data
                scheduler (obj): protect us from time limit exceeding
                algo (cls): to be instantiated with given config(s)
                opt_records (list): given config(s)
                learn_from_scratch (bool): restore from the ckpt(s) or 
                                           random initialize the parameters
            Returns: predictions (np.ndarray)
        """

        logger.info('Final algo is: %s', algo)
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
                    # naive version: enhance the model by train with the valid/whole data part
                    i = 0
                    # todo: add some other heuristic method to set the small_epoch,
                    #  e.g., the average epochs of the stopper
                    while not scheduler.should_stop(SAFE_FRAC) and i < FINE_TUNE_EPOCH:
                        # model.train(data, valid_mask)  # fine-tune on the un-seen valid set
                        model.train(data, data.train_mask)  # fine-tune on the whole data
                        i += 1
                    logger.info("Fine-tune when cv, fine tune epoch: {}/{}".format(i, FINE_TUNE_EPOCH))
                cur_valid_part_idx += 1
            if len(part_logits) == 0:
                logger.warn("have not completed even one training course")
                logits = model.pred(data, make_decision=False)
                part_logits.append(logits.cpu().numpy())
            logger.info("ensemble {} models".format(len(part_logits)))
            pred = np.argmax(np.mean(np.stack(part_logits), 0), -1).flatten()
            return pred
        elif self._training_strategy == 'naive':
            if len(opt_records) == 1:
                # just train a model with the optimal config on the whole labeled samples
                pass
            else:
                # besides of the diversity of inductive biases, we also
                # exploit the diversity of their training data
                parts = divide_data(data, CV_NUM_FOLD*[10/CV_NUM_FOLD], device)
            preds = list()
            weights = list()
            for i, opt_record in enumerate(opt_records):
                model = algo(n_class, num_features, device, opt_record[0], non_hpo_config)
                if not learn_from_scratch:
                    model.load_model(opt_record[1])
                if len(opt_records) == 1:
                    train_mask = data.train_mask
                    valid_mask = None
                else:
                    train_mask = torch.sum(
                        torch.stack([m for j, m in enumerate(parts) if j != (i%CV_NUM_FOLD)]), 0).type(torch.bool)
                    valid_mask = parts[i%CV_NUM_FOLD]
                self._ensembler_early_stopper.reset()
                while not scheduler.should_stop(SAFE_FRAC):
                    train_info = model.train(data, train_mask)
                    if valid_mask is not None:
                        valid_info = model.valid(data, valid_mask)
                    else:
                        # currently, this only cooperates with fixed #epochs
                        valid_info = None
                    if self._ensembler_early_stopper.should_early_stop(train_info, valid_info) and \
                       (not learn_from_scratch or self._ensembler_early_stopper.get_cur_step() >= opt_record[3]):
                        activation = model.pred(data, make_decision=False)
                        pr = F.softmax(activation)
                        preds.append(pr)
                        weights.append(opt_record[2])
                        break
                logger.info("the {}-th final model traverses the whole \
                             training data for {} epochs".format(i, self._ensembler_early_stopper.get_cur_step()))
            if len(preds) == 0:
                logger.warn("have not completed even one training course")
                activation = model.pred(data, make_decision=False)
                pr = F.softmax(activation)
                preds.append(pr)
                weights.append(opt_record[2])
            # weighted average the probabilities
            weights = torch.tensor(np.asarray(weights), dtype=torch.float32).to(device)
            weights = weights / torch.sum(weights)
            logger.info("average {} models with their validation performances {}".format(len(preds), weights))
            single_shape = preds[0].shape
            preds = torch.reshape(torch.stack(preds), weights.shape+single_shape)
            preds = torch.mean(torch.reshape(weights, weights.shape+(1, 1)) * preds, 0)
            pred = torch.argmax(preds, -1).cpu().numpy().flatten()
            return pred
        else:
            # TO DO: provide other strategies
            pass
