from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .gcn_algo import GNNAlgo, FocalLoss
from .gnns import *


from spaces import Categoric, Numeric


class SGCNAlgo(GNNAlgo):
    hyperparam_space = dict(
        num_layers=Categoric(list(range(2,5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        hidden_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4)
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config):
        self._device = device
        self._num_class = num_class
        self.model = SGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num


class SplineGCNAlgo(GNNAlgo):

    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 100], None, 16),
        hiden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Categoric([0.0, 0.2, 0.5], None, 0.0),
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        loss_type=Categoric(["focal_loss", "ce_loss"], None, "focal_loss"),
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config,
                 non_hpo_config,
                 ):
        self._device = device
        self._num_class = num_class
        self.model = SplineGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("dim", 1),
            config.get("kernel_size", 2), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.loss_type = config.get("loss_type", "focal_loss")
        self.fl_loss = FocalLoss(config.get("gamma", 2), non_hpo_config.get("label_alpha", []), device)


class SplineGCN_APPNPAlgo(GNNAlgo):
    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 100], None, 16),
        hiden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Categoric([0.0, 0.2, 0.5], None, 0.0),
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        loss_type=Categoric(["focal_loss", "ce_loss"], None, "focal_loss"),
        iter_k=Categoric([10, 20, 40], None, 10),
        tele_prob=Categoric([0.2, 0.5, 0.7], None, 0.5),
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config,
                 focal_loss=False):
        self._device = device
        self._num_class = num_class
        self.model = SplineGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("dim", 1),
            config.get("kernel_size", 2), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.focal_loss = focal_loss



