from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from early_stoppers import Stopper

logger = logging.getLogger('code_submission')

# current implementation: run WINDOW_SIZE steps at least
WINDOW_SIZE = 20


class NonImprovementStopper(Stopper):

    def __init__(self, min_step=40, max_step=800):
        self._min_step = max(min_step, WINDOW_SIZE)
        self._max_step = max_step
        self._step_with_max_performance = -1
        self._max_performance = -float('inf')
        super(NonImprovementStopper, self).__init__()

    def should_early_stop(self, train_info, valid_info):
        self._cur_step += 1
        cur_performance = -valid_info['loss']
        # cur_performance = valid_info['accuracy']
        if cur_performance > self._max_performance:
            self._step_with_max_performance = self._cur_step
            self._max_performance = cur_performance
        if self._cur_step > self._min_step and \
                (self._cur_step-self._step_with_max_performance) >= WINDOW_SIZE:
            logger.info("early stop at {} epoch".format(self._cur_step))
            return True
        return self._cur_step >= self._max_step

    def reset(self):
        self._cur_step = 0
        self._step_with_max_performance = -1
        self._max_performance = -float('inf')
