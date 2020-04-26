from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from early_stoppers.stopper import Stopper
from early_stoppers.constant_stopper import ConstantStopper


agents = dict(
    Stopper=Stopper,
    ConstantStopper=ConstantStopper)

__all__ = ["Stopper", "ConstantStopper"]
