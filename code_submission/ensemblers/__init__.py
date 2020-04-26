from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ensemblers.ensembler import Ensembler
from ensemblers.greedy_strategy import GreedyStrategy


agents = dict(
    Ensembler=Ensembler,
    GreedyStrategy=GreedyStrategy)

__all__ = ["Ensembler", "GreedyStrategy"]
