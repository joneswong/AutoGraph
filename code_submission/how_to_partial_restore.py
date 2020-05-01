from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from algorithms import GCNAlgo


def main():
    config = dict(
        num_layers=1,
        hidden=16,
        dropout_rate=0.75,
        lr=5e-3,
        weight_decay=5e-4)
    pretrained = GCNAlgo(2, 256, 'cuda:0', config)
    for k,v in pretrained.model.state_dict().items():
        print("{}: {}".format(k, v))
    print("=========================================")
    pretrained.save_model("pretrained.pt")

    # shallow to deep
    new_config = copy.deepcopy(config)
    new_config["num_layers"] = 2
    new = GCNAlgo(2, 256, 'cuda:0', new_config)
    for k,v in new.model.state_dict().items():
        print("{}: {}".format(k, v))
    print("=========================================")
    new.load_model("pretrained.pt", strict=False)
    for k,v in new.model.state_dict().items():
        print("{}: {}".format(k, v))
    print("=========================================")
    new .save_model("new.pt")

    # deep to shallow
    new = GCNAlgo(2, 256, 'cuda:0', config)
    for k,v in new.model.state_dict().items():
        print("{}: {}".format(k, v))
    print("=========================================")
    new.load_model("new.pt", filter_out=True)
    for k,v in new.model.state_dict().items():
        print("{}: {}".format(k, v))
    print("=========================================")


if __name__ == "__main__":
    main()
