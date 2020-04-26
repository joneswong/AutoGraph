from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ensemblers import Ensembler

class GreedyStrategy(Ensembler):

    def __init__(self, finetune=False):
        super(GreedyStrategy, self).__init__(finetune)

    def boost(self, data, results, cls):
        sorted_results = sorted(results, key=lambda x:x[2]["accuracy"])
        optimal = sorted_results[-1]
        if self._finetune:
            # (jones.wz) TO DO: implement
            pass
        else:
            model = cls(**optimal[0])
            model.load_state_dict(torch.load(optimal[1]))
        return model
