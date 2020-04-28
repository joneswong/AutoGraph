from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spaces.space import Space
from spaces.categoric import Categoric
from spaces.numeric import Numeric


agents = dict(
    Space=Space,
    Categoric=Categoric,
    Numeric=Numeric)

__all__ = ["Space", "Categoric", "Numeric"]
