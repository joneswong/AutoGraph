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

    def __repr__(self):
        space_desc = "shape is %s, dtype is %s, low and high is [%s, %s], default_value is %s." % \
                     (str(self.shape), str(self.dtype), str(self.low), str(self.high), str(self.default_value))
        return space_desc
