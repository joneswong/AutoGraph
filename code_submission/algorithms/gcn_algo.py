from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge

from spaces import Categoric, Numeric


class GCN(torch.nn.Module):

    def __init__(self,
                 num_class,
                 features_num,
                 num_layers=2,
                 hidden=16):

        super(GCN, self).__init__()
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=0.5, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCNAlgo(object):
    """Encapsulate the torch.Module and the train&valid&test routines"""

    hyperparam_space = dict(
        num_layers=Numeric((), np.int32, 1, 4, 2),
        hidden=Numeric((), np.int32, 4, 32, 16),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4))

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config):

        self._num_class = num_class
        self._features_num = features_num
        self._device = device
        self.model = GCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        
    def train(self, data):
        self.model.train()
        self._optimizer.zero_grad()
        loss = F.nll_loss(
            self.model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self._optimizer.step()
        # We may need values other than logloss for making better decisions on
        # when to stop the training course
        return {"logloss": loss}

    def valid(self, data):
        self.model.eval()
        # TO DO: implement
        return {"logloss": .0, "accuracy": 1.0}
    
    def pred(self, data):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data)[data.test_mask].max(1)[1]
        return pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self._device))
