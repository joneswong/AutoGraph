from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

from early_stoppers import Stopper

logger = logging.getLogger('code_submission')

# current implementation: run WINDOW_SIZE steps at least
WINDOW_SIZE = 80
MAX_T = 1e5
START_MUL = 1.2
END_MUL = 1.0


class AdaptiveWeightStopper(Stopper):

    def __init__(self, min_step=160, max_step=800):
        self._min_step = max(min_step, WINDOW_SIZE)
        self._max_step = max_step
        self._step_with_max_performance = -1
        self._max_performance = -float('inf')
        self._cur_T = 1.0
        self._performance_list = []
        self.a = (END_MUL - START_MUL) / (math.log(MAX_T) / math.log(10))
        self.b = START_MUL
        super(AdaptiveWeightStopper, self).__init__()

    def should_early_stop(self, train_info, valid_info):
        self._cur_step += 1
        cur_performance = valid_info['accuracy']
        self._performance_list.append(cur_performance)
        if cur_performance > self._max_performance:
            self._step_with_max_performance = self._cur_step
            self._max_performance = cur_performance
        if self._cur_step > self._min_step and \
                (self._cur_step-self._step_with_max_performance) >= WINDOW_SIZE:
            logger.info("early stop at {} epoch".format(self._cur_step))
            return True
        return self._cur_step >= self._max_step

    def should_log(self, train_info, valid_info):
        return valid_info['accuracy'] > self._max_performance

    def get_T(self):
        if len(self._performance_list) != 0:
            should_improve_minority = (self._performance_list[-1] >= 0.8) and ((self._max_performance - self._performance_list[-1]) <= 0.03)
            if should_improve_minority:
                # self._cur_T = min(self._cur_T * 1.2, 100000.0)
                self._cur_T = (self.a * math.log(self._cur_T) / math.log(10) + self.b) * self._cur_T
            else:
                self._cur_T = max(self._cur_T / 2.0, 1.0)
        return self._cur_T

    def reset(self):
        self._cur_step = 0
        self._step_with_max_performance = -1
        self._max_performance = -float('inf')
        self._cur_T = 1.0
        self._performance_list = []
