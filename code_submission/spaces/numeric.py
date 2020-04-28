from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spaces import Space


class Numeric(Space):

    def __init__(self, shape, dtype, low, high, default_value):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.high = high
        super(Numeric, self).__init__(default_value)
