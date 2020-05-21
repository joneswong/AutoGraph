from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from early_stoppers.stopper import Stopper
from early_stoppers.constant_stopper import ConstantStopper
from early_stoppers.stable_stopper import StableStopper
from early_stoppers.memory_stopper import MemoryStopper
from early_stoppers.empirical_stopper import EmpiricalStopper
from early_stoppers.non_improvement_stopper import NonImprovementStopper
from early_stoppers.adaptive_weight_stopper import AdaptiveWeightStopper


agents = dict(
    Stopper=Stopper,
    ConstantStopper=ConstantStopper,
    StableStopper=StableStopper,
    MemoryStopper=MemoryStopper,
    EmpiricalStopper=EmpiricalStopper,
    NonImprovementStopper=NonImprovementStopper,
    AdaptiveWeightStopper=AdaptiveWeightStopper
)

__all__ = ["Stopper", "ConstantStopper", "StableStopper", "MemoryStopper", "EmpiricalStopper", "NonImprovementStopper", "AdaptiveWeightStopper"]
