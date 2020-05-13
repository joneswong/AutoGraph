from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.stats import truncnorm
import numpy as np
import random
import copy

from schedulers import Scheduler
from spaces import Categoric, Numeric
from utils import get_performance


class GeneticOptimizer(Scheduler):
    """Apply Bayesian optimization for HPO"""

    def __init__(self,
                 hyperparam_space,
                 early_stopper,
                 ensembler,
                 max_population=3):

        super(GeneticOptimizer, self).__init__(
            hyperparam_space, early_stopper, ensembler)

        for key, value in self._hyperparam_space.items():
            if isinstance(value, Categoric):
                sorted_categories = sorted(value.categories)
                self._hyperparam_space[key].categories = sorted_categories
            elif isinstance(value, Numeric):
                assert value.low <= value.high
            else:
                raise NotImplementedError

        self._population = []
        self._max_population = max_population
        self._cur_config = self.get_default()

    def get_next_config(self):
        self._early_stopper.reset()

        if len(self._results) != 0:
            self.gen_next_population()
            crossover_config = self.crossover()
            self._cur_config = self.mutation(crossover_config)

        return self._cur_config

    def crossover(self):
        s1 = random.randint(0, len(self._population)-1)
        s2 = random.randint(0, len(self._population)-1)
        if s1 == s2:
            h_son = self._population[s1][0]
        else:
            h1 = self._population[s1][0]
            h2 = self._population[s2][0]
            h_son = {}
            for key in h1.keys():
                h_son[key] = h1[key] if random.random() <= 0.5 else h2[key]
        return h_son

    def mutation(self, h):
        for key, value in h.items():
            space = self._hyperparam_space[key]
            if isinstance(space, Categoric):
                cur_index = list(space.categories).index(value)
                if type(value) is int or type(value) is float:
                    clip_a, clip_b, mean, std = 0, len(space.categories)-1e-10, cur_index+0.5, len(space.categories)/6
                    a, b = (clip_a - mean) / std, (clip_b - mean) / std
                    new_index = int(truncnorm.rvs(a, b, mean, std, random_state=random.randint(0, 1e5)))
                else:
                    new_index = random.randint(0, len(space.categories)-1)
                h[key] = space.categories[new_index]
            elif isinstance(space, Numeric):
                if space.high-space.low != 0:
                    clip_a, clip_b, mean, std = space.low, space.high, value, (space.high-space.low)/6
                    a, b = (clip_a - mean) / std, (clip_b - mean) / std
                    new_value = truncnorm.rvs(a, b, mean, std, random_state=random.randint(0, 1e5))
                    h[key] = new_value
            else:
                raise NotImplementedError
        return h

    def gen_next_population(self):
        performance = self._results[-1][2]['accuracy']
        if len(self._population) < self._max_population:
            self._population.append((copy.deepcopy(self._cur_config), performance))
        else:
            replaced_index = int(np.argmin([item[1] for item in self._population]))
            self._population[replaced_index] = (copy.deepcopy(self._cur_config), performance)
