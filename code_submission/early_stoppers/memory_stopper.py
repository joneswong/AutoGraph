from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging

from early_stoppers import Stopper
from utils import get_performance

logger = logging.getLogger('code_submission')

# current implementation: run WINDOW_SIZE steps at least
WINDOW_SIZE = 20


class MemoryStopper(Stopper):

    def __init__(self, min_step=30, max_step=500):
        self._min_step = max(min_step, WINDOW_SIZE)
        self._max_step = max_step

        # self.performance_memory -> index: step, value: (average performance, count)
        self.performance_memory = [(0.0, 0) for i in range(max_step+1)]
        self.performance_windows = [None for i in range(WINDOW_SIZE)]
        self.index = 0
        self.max_acc = -float('inf')

        super(MemoryStopper, self).__init__()

    def should_early_stop(self, train_info, valid_info):
        self._cur_step += 1
        self.max_acc = max(self.max_acc, valid_info['accuracy'])
        # performance = get_performance(valid_info)
        performance = -valid_info['loss']
        self.performance_windows[self.index] = performance
        self.index = (self.index + 1) % WINDOW_SIZE

        if self._cur_step >= self._min_step:
            # average over window to reduce variance
            ave_performance_over_window = np.mean(self.performance_windows)
            ave_performance_over_trials = self.performance_memory[self._cur_step][0]
            count = self.performance_memory[self._cur_step][1]
            self.performance_memory[self._cur_step] = ((ave_performance_over_trials*count+ave_performance_over_window)/(count+1), count+1)
            if ave_performance_over_window < self.performance_memory[self._cur_step][0]:
                logger.info("early stop at {} epoch".format(self._cur_step))
                return True

        return self._cur_step >= self._max_step

    def should_log(self, train_info, valid_info):
        return valid_info['accuracy'] > self.max_acc

    def reset(self):
        self._cur_step = 0
        self.index = 0
        self.performance_windows = [None for i in range(WINDOW_SIZE)]
        self.max_acc = -float('inf')
