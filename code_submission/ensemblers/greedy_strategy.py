from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ensemblers import Ensembler
from utils import get_performance

class GreedyStrategy(Ensembler):

    def __init__(self, finetune=False):
        super(GreedyStrategy, self).__init__(finetune)

    def boost(self, n_class, features_num, device, data, results, algo):
        sorted_results = sorted(results, key=lambda x: x[2]['accuracy'])
        optimal = sorted_results[-1]
        if self._finetune:
            # TO DO: implement
            pass
        else:
            config = optimal[0]
            model = algo(n_class, features_num, device, config)
            model.load_model(optimal[1])
        return model


