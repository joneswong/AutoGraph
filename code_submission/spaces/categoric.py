from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spaces import Space


class Categoric(Space):

    def __init__(self, categories, subspaces, default_value):
        self.categories = categories
        self.subspaces = subspaces
        super(Categoric, self).__init__(default_value)
