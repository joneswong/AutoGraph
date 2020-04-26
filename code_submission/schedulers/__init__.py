from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from schedulers.scheduler import Scheduler
from schedulers.grid_search import GridSearcher
from schedulers.bayesian_optimization import BayesianOptimizer


agents = dict(
    Scheduler=Scheduler,
    GridSearcher=GridSearcher,
    BayesianOptimizer=BayesianOptimizer)

__all__ = ["Scheduler", "GridSearcher", "BayesianOptimizer"]
