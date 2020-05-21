from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from early_stoppers import Stopper
from utils import get_performance

logger = logging.getLogger('code_submission')

# be very careful when setting STOP_STD
# (highly related to the adopted performance evaluation function)
# too large -> stop too fast, too small -> do not stop early
STOP_STD = 0.0001
# current implementation: run WINDOW_SIZE steps at least
WINDOW_SIZE = 10


class StableStopper(Stopper):

    def __init__(self, max_step=800):

        self._max_step = max_step
        self.performance_windows = [None for i in range(WINDOW_SIZE)]
        self.index = 0
        super(StableStopper, self).__init__()

    def should_early_stop(self, train_info, valid_info):
        self._cur_step += 1
        # self.performance_windows[self.index] = get_performance(valid_info)
        self.performance_windows[self.index] = -valid_info['loss']
        self.index = (self.index + 1) % WINDOW_SIZE
        if self.performance_windows[self.index] is not None and \
                np.std(self.performance_windows) < STOP_STD:
            logger.info("early stop at {} epoch".format(self._cur_step))
            return True
        return self._cur_step >= self._max_step

    def reset(self):
        self._cur_step = 0
        self.index = 0
        self.performance_windows = [None for i in range(WINDOW_SIZE)]
