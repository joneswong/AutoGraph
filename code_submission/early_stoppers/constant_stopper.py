from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from early_stoppers import Stopper


class ConstantStopper(Stopper):

    def __init__(self, max_step=800):

        self._max_step = max_step
        super(ConstantStopper, self).__init__()

    def should_early_stop(self, train_info, valid_info):
        self._cur_step += 1
        return self._cur_step >= self._max_step
