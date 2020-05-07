from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from schedulers.scheduler import Scheduler
from schedulers.grid_search import GridSearcher
from schedulers.bayesian_optimization import BayesianOptimizer
from schedulers.genetic_optimization import GeneticOptimizer


agents = dict(
    Scheduler=Scheduler,
    GridSearcher=GridSearcher,
    BayesianOptimizer=BayesianOptimizer,
    GeneticOptimizer=GeneticOptimizer
)

__all__ = ["Scheduler", "GridSearcher", "BayesianOptimizer", "GeneticOptimizer"]
