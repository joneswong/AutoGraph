from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spaces import Space


class Categoric(Space):

    def __init__(self, categories, subspaces, default_value):
        self.categories = categories
        self.subspaces = subspaces
        super(Categoric, self).__init__(default_value)

    def desc(self):
        space_desc = "categories is %s, subspaces is %s, default_value is %s." % \
                     (str(self.categories), str(self.subspaces), str(self.default_value))
        return space_desc
