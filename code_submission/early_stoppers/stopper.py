from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Stopper(object):

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        self._cur_step = 0

    def should_early_stop(self, train_info, valid_info):
        """Decide whether to do an earlystop
        Arguments:
            train_info (dict): logloss, ...
            valid_info (dict): logloss, accuracy, ...
        Returns: True or False accoridng to the decision
        """
        raise NotImplementedError

    def get_cur_step(self):
        return self._cur_step
