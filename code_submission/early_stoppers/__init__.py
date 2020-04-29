from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from early_stoppers.stopper import Stopper
from early_stoppers.constant_stopper import ConstantStopper
from early_stoppers.stable_stopper import StableStopper


agents = dict(
    Stopper=Stopper,
    ConstantStopper=ConstantStopper,
    StableStopper=StableStopper)

__all__ = ["Stopper", "ConstantStopper", "StableStopper"]
