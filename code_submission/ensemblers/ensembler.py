from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Ensembler(object):

    def __init__(self, finetune=False, *args, **kwargs):
        self._finetune = finetune

    def boost(self, data, results):
        raise NotImplementedError
