from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge
from sklearn.metrics import accuracy_score

from spaces import Categoric, Numeric

from torch_geometric.utils.dropout import dropout_adj


class GCN(torch.nn.Module):
    def __init__(self,
                 num_class,
                 features_num,
                 num_layers=2,
                 hidden=16,
                 hidden_droprate=0.5, edge_droprate=0.0):

        super(GCN, self).__init__()
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
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
        x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GNNAlgo(object):
    """Encapsulate the torch.Module and the train&valid&test routines"""

    hyperparam_space = dict()

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config):

        self._num_class = num_class
        self._features_num = features_num
        self._device = device
        self._num_class = num_class
        self.model = GCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16), config.get("hidden_droprate", 0.5)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        
    def train(self, data):
        self.model.train()
        self._optimizer.zero_grad()
        loss = F.nll_loss(
            self.model(data)[data_mask], data.y[data_mask])
        loss.backward()
        self._optimizer.step()
        # We may need values other than logloss for making better decisions on
        # when to stop the training course
        return {"logloss": loss.item()}

    def valid(self, data, data_mask):
        self.model.eval()
        with torch.no_grad():
            validation_output = self.model(data)[data_mask]
            validation_pre = validation_output.max(1)[1]
            validation_truth = data.y[data_mask]
            logloss = F.nll_loss(validation_output, validation_truth)

        cpu = torch.device('cpu')
        accuracy = accuracy_score(validation_truth.to(cpu), validation_pre.to(cpu))
        return {"logloss": logloss.item(), "accuracy": accuracy}
    
    def pred(self, data, make_decision=True):
        self.model.eval()
        with torch.no_grad():
            if make_decision:
                pred = self.model(data)[data.test_mask].max(1)[1]
            else:
                pred = self.model(data)[data.test_mask]
        return pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, filter_out=False, strict=True):
        prev_state_dict = torch.load(path, map_location=self._device)
        if filter_out:
            prev_state_dict = {k: v for k, v in prev_state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(prev_state_dict, strict=strict)


class GCNAlgo(GNNAlgo):

    hyperparam_space = dict(
        num_layers=Categoric(list(range(2,5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        hidden_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4),
        edge_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config):
        self._device = device
        self._num_class = num_class
        self.model = GCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16), config.get("hidden_droprate", config.get("edge_droprate", 0.0)).to(device))
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num

