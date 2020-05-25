from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch
import torch.nn.functional as F

from early_stoppers import ConstantStopper
from utils import divide_data, divide_data_label_wise, calculate_config_dist

logger = logging.getLogger('code_submission')

CV_NUM_FOLD=5
SAFE_FRAC=0.95
FINE_TUNE_EPOCH=50
FINE_TUNE_WHEN_CV=False


class Ensembler(object):

    def __init__(self,
                 early_stopper,
                 config_selection='greedy',
                 training_strategy='cv',
                 return_best=False,
                 *args,
                 **kwargs):

        self._ensembler_early_stopper = early_stopper
        self._config_selection = config_selection
        self._training_strategy = training_strategy
        self._return_best = return_best

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
        sorted_results = sorted(results, key=lambda x: x[2]['accuracy'])
        self.train_over_all_data = (len(sorted_results) == 1 and sorted_results[0][3] < 200)

        if self._config_selection == 'greedy':
            # choose the best one
            optimal = sorted_results[-1]
            return [optimal]
        elif self._config_selection.startswith("top"):
            if self._config_selection.endswith("lo"):
                # in the form "top(\d+)lo" meaning to
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
                                sorted_results[i])
                            if dist > cur_max_dist:
                                cur_max_dist = dist
                                picked_idx = i
                    picked.append(sorted_results[picked_idx])
                    last_selected = picked_idx
                    flags[picked_idx] = 0
                logger.info("choosed {}".format(','.join([str(v) for v in flags])))
                return picked
            else:
                # in the form "top(\d+)" meaning to choose the topK
                K = min(len(sorted_results), int(self._config_selection[3:]))
                considered = sorted_results[-K:]
                considered.reverse()
                return considered
        elif self._config_selection == 'auto':
            sorted_results.reverse()
            reversed_sorted_results = sorted_results

            # find top k configs automatically
            top_k = 0
            best_performance = reversed_sorted_results[0][2]['accuracy']
            # pre_performance = best_performance
            for i in range(len(reversed_sorted_results)):
                cur_performance = reversed_sorted_results[i][2]['accuracy']
                if best_performance-cur_performance > 0.1: # or (pre_performance-cur_performance)>0.03
                    top_k = i
                    break
                top_k = i + 1
                # pre_performance = cur_performance
            top_k = min(10, top_k)
            return reversed_sorted_results[:top_k]
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
                 non_hpo_config=dict(),
                 train_y=None):
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
            logger.info("searched opt_config is {}.".format(opt_record))

        if self._training_strategy == 'cv':
            preds = self.cv_ensembler(opt_records, data, device, n_class, num_features,
                                      scheduler, algo, learn_from_scratch, non_hpo_config)
            pred = torch.argmax(torch.mean(preds, 0), -1).flatten()
            return pred.cpu().numpy()
        elif self._training_strategy == 'naive':
            weighted_preds = self.naive_ensembler(opt_records, data, device, n_class, num_features,
                                 scheduler, algo, learn_from_scratch, non_hpo_config, train_y)
            pred = torch.argmax(torch.mean(weighted_preds, 0), -1).flatten()
            return pred.cpu().numpy()
        elif self._training_strategy == 'hpo_trials':
            weighted_preds = self.hpo_history_ensembler(opt_records, device)
            pred = torch.argmax(torch.mean(weighted_preds, 0), -1).flatten()
            return pred.cpu().numpy()
        elif self._training_strategy == 'hybrid':
            weighted_preds_of_hpo_trials = self.hpo_history_ensembler(opt_records, device)
            weighted_preds_of_naive_results = self.naive_ensembler(opt_records, data, device, n_class, num_features,
                                                                   scheduler, algo, learn_from_scratch, non_hpo_config, train_y)
            weighted_preds = torch.cat([weighted_preds_of_hpo_trials, weighted_preds_of_naive_results], 0)
            pred = torch.argmax(torch.mean(weighted_preds, 0), -1).flatten()
            return pred.cpu().numpy()
        else:
            # TO DO: provide other strategies
            pass

    def cv_ensembler(self, opt_records, data, device, n_class, num_features, scheduler, algo, learn_from_scratch, non_hpo_config):
        opt_record = opt_records[0]
        if self.train_over_all_data:
            global CV_NUM_FOLD
            CV_NUM_FOLD = 1
        parts = divide_data(data, CV_NUM_FOLD * [10 / CV_NUM_FOLD], device)
        part_logits = list()
        cur_valid_part_idx = 0
        while (not scheduler.should_stop(SAFE_FRAC)): # and (cur_valid_part_idx < CV_NUM_FOLD):
            model = algo(n_class, num_features, device, opt_record[0], non_hpo_config)
            if not learn_from_scratch:
                model.load_model(opt_record[1])
            if not self.train_over_all_data:
                train_mask = torch.sum(torch.stack([m for i, m in enumerate(parts) if i != cur_valid_part_idx]), 0).type(torch.bool)
                valid_mask = parts[cur_valid_part_idx]
            else:
                train_mask = parts[0]
                valid_mask = parts[0]
            self._ensembler_early_stopper.reset()
            while not scheduler.should_stop(SAFE_FRAC):
                train_info = model.train(data, train_mask)
                valid_info = model.valid(data, valid_mask)
                if self._ensembler_early_stopper.should_early_stop(train_info, valid_info) or self.train_over_all_data:
                    logits = model.pred(data, make_decision=False)
                    part_logits.append(logits.cpu().numpy())
                    if self._ensembler_early_stopper.should_early_stop(train_info, valid_info):
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
            cur_valid_part_idx = (cur_valid_part_idx + 1) % CV_NUM_FOLD
            if cur_valid_part_idx == 0:
                parts = divide_data(data, CV_NUM_FOLD * [10 / CV_NUM_FOLD], device)
        if len(part_logits) == 0:
            logger.warn("have not completed even one training course")
            logits = model.pred(data, make_decision=False)
            part_logits.append(logits)
        logger.info("ensemble {} models".format(len(part_logits)))
        preds = F.softmax(torch.stack(part_logits), -1)
        return preds

    def naive_ensembler(self, opt_records, data, device, n_class, num_features, scheduler, algo, learn_from_scratch, non_hpo_config, train_y):
        if self.train_over_all_data:
            # limited time for training even one model
            pass
        else:
            # besides of the diversity of inductive biases, we also
            # exploit the diversity of their training data
            parts = divide_data_label_wise(data, CV_NUM_FOLD * [10 / CV_NUM_FOLD], device, n_class, train_y)
        preds = list()
        config_weights = list()
        finetuned_model_weights = list()
        cur_index = 0
        tmp_results = None
        tmp_valid_info = None
        while not scheduler.should_stop(SAFE_FRAC):
            opt_record = opt_records[cur_index % len(opt_records)]
            model = algo(n_class, num_features, device, opt_record[0], non_hpo_config)
            if not learn_from_scratch:
                model.load_model(opt_record[1])
            if self.train_over_all_data:
                train_mask = data.train_mask
                valid_mask = data.train_mask
            else:
                train_mask = torch.sum(
                    torch.stack([m for j, m in enumerate(parts) if j != (cur_index % CV_NUM_FOLD)]), 0).type(torch.bool)
                valid_mask = parts[cur_index % CV_NUM_FOLD]
            self._ensembler_early_stopper.reset()
            while not scheduler.should_stop(SAFE_FRAC):
                train_info = model.train(data, train_mask)
                valid_info = model.valid(data, valid_mask)
                if self._return_best and self._ensembler_early_stopper.should_log(train_info, valid_info):
                    tmp_results = model.pred(data, make_decision=False)
                    tmp_valid_info = valid_info
                if self._ensembler_early_stopper.should_early_stop(train_info, valid_info) or self.train_over_all_data:
                    if self._return_best:
                        pr = F.softmax(tmp_results)
                        preds.append(pr)
                        config_weights.append(opt_record[2]['accuracy'])
                        finetuned_model_weights.append(tmp_valid_info['accuracy'])
                    else:
                        activation = model.pred(data, make_decision=False)
                        pr = F.softmax(activation)
                        preds.append(pr)
                        config_weights.append(opt_record[2]['accuracy'])
                        finetuned_model_weights.append(valid_info['accuracy'])
                    if self._ensembler_early_stopper.should_early_stop(train_info, valid_info):
                        break
            logger.info("the {}-th final model traverses the whole \
                         training data for {} epochs".format(cur_index, self._ensembler_early_stopper.get_cur_step()))
            cur_index = cur_index + 1
            if cur_index % CV_NUM_FOLD == 0:
                if len(opt_records) == 1:
                    pass
                else:
                    parts = divide_data_label_wise(data, CV_NUM_FOLD * [10 / CV_NUM_FOLD], device, n_class, train_y)
        if len(preds) == 0:
            logger.warn("have not completed even one training course")
            activation = model.pred(data, make_decision=False)
            pr = F.softmax(activation)
            preds.append(pr)
            config_weights.append(opt_record[2]['accuracy'])
        # weighted average the probabilities
        config_weights = torch.tensor(np.asarray(config_weights), dtype=torch.float32).to(device)
        # config_weights = config_weights / torch.sum(config_weights)
        if len(finetuned_model_weights):
            finetuned_model_weights = torch.tensor(np.asarray(finetuned_model_weights), dtype=torch.float32).to(device)
            # finetuned_model_weights = finetuned_model_weights / torch.sum(finetuned_model_weights)
            weights = 0.5 * (config_weights + finetuned_model_weights)
        else:
            weights = config_weights
        logger.info("average {} models with their validation performances {}".format(len(preds), weights))
        single_shape = preds[0].shape
        preds = torch.reshape(torch.stack(preds), weights.shape + single_shape)
        weighted_preds = torch.reshape(weights, weights.shape + (1, 1)) * preds
        return weighted_preds

    def hpo_history_ensembler(self, opt_records, device):
        part_logits = []
        for i in range(len(opt_records)):
            path = opt_records[i][4]
            logits = torch.load(path)['test_results']
            part_logits.append(logits)
        logger.info("ensemble {} models".format(len(part_logits)))
        weights = torch.tensor(
            np.array([[[item[2]['accuracy']]] for item in opt_records]),
            dtype=torch.float32).to(device)
        weights_preds = weights * F.softmax(torch.stack(part_logits), -1)
        return weights_preds
