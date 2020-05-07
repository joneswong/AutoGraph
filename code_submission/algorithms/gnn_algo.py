from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.nn import Linear, functional as F
from torch_geometric.nn import JumpingKnowledge, SGConv, SplineConv, APPNP
from .gcn_algo import GNNAlgo, FocalLoss
from .gnn_tricks import GraphSizeNorm


from spaces import Categoric, Numeric

from torch_geometric.utils.dropout import dropout_adj


class SplineGCN(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, droprate=0.5, dim=1, kernel_size=2,
                 edge_droprate=0.0, fea_norm="no_norm"):
        super(SplineGCN, self).__init__()
        self.droprate = droprate
        self.edge_droprate = edge_droprate
        if fea_norm == "no_norm":
            self.fea_norm_layer = None
        elif fea_norm == "graph_size_norm":
            self.fea_norm_layer = GraphSizeNorm()
        else:
            raise ValueError("your fea_norm is un-defined: %s") % fea_norm

        self.convs = torch.nn.ModuleList()
        self.convs.append(SplineConv(features_num, hidden, dim, kernel_size))
        for i in range(num_layers - 2):
            self.convs.append(SplineConv(hidden, hidden, dim, kernel_size))
        self.convs.append(SplineConv(hidden, num_class, dim, kernel_size))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for conv in self.convs:
            x = x if self.fea_norm_layer is None else self.fea_norm_layer(x)
            x = F.dropout(x, p=self.droprate, training=self.training)
            x = F.elu(conv(x, edge_index, edge_weight))
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




class SplineGCN_APPNP(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, droprate=0.5, dim=1, kernel_size=2,
                 edge_droprate=0.0, fea_norm="no_norm", K=20, alpha=0.5):
        super(SplineGCN, self).__init__()
        self.droprate = droprate
        self.edge_droprate = edge_droprate
        if fea_norm == "no_norm":
            self.fea_norm_layer = None
        elif fea_norm == "graph_size_norm":
            self.fea_norm_layer = GraphSizeNorm()
        else:
            raise ValueError("your fea_norm is un-defined: %s") % fea_norm

        self.convs = torch.nn.ModuleList()
        self.convs.append(SplineConv(features_num, hidden, dim, kernel_size))
        for i in range(num_layers - 2):
            self.convs.append(SplineConv(hidden, hidden, dim, kernel_size))
        self.convs.append(SplineConv(hidden, num_class, dim, kernel_size))

        self.appnp = APPNP(K, alpha)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for conv in self.convs:
            x = x if self.fea_norm_layer is None else self.fea_norm_layer(x)
            x = F.dropout(x, p=self.droprate, training=self.training)
            x = F.elu(conv(x, edge_index, edge_weight))
        x = self.appnp(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SGCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, hidden_droprate=0.5, edge_droprate=0.0):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SGConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.hidden_droprate = hidden_droprate
        self.edge_droprate = edge_droprate

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


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
        num_layers=Categoric(list(range(1,5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        hidden_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Numeric((), np.float32, 0.0, 0.0, 0.0),
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        loss_type=Categoric(["focal_loss", "ce_loss"], None, "focal_loss"),
        #gamma=Categoric([0.2, 0.5, 1.0, 2.0, 4.0], None, 2.0),
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
        num_layers=Categoric(list(range(1,5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        hidden_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Numeric((), np.float32, 0.0, 0.0, 0.0),
        # todo (daoyuan): add pair_norm and batch_norm
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "graph_size_norm"),
        iter_k=Categoric([10, 20, 40], None, 10),
        tele_prob=Numeric((), np.float32, 0.2, 0.7, 0.5),
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



