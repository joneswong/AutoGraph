from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, SGConv, SplineConv
from torch_geometric.data import Data

from schedulers import Scheduler
from early_stoppers import ConstantStopper
from ensemblers import GreedyStrategy

# TODO (daoyuan): we can set this fraction value adaptively by recording or estimating the prediction time
FRAC_FOR_SEARCH = 0.85


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(1234)


class SplineGCN(torch.nn.Module):
    config_desc = dict(num_layers=[1, 2], hidden=[4, 32], dim =[0, 1], kernel_size=[2, 4])
    default_config = dict(num_layers=2, hidden=16, features_num=16, dim=1, kernel_size=2)

    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, dim=1, kernel_size=2):
        super(SplineGCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SplineConv(features_num, hidden, dim, kernel_size))
        self.convs.append(SplineConv(hidden, num_class, dim, kernel_size))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(conv(x, edge_index, edge_weight))
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class SGCN(torch.nn.Module):
    config_desc = dict(num_layers=[1, 2], hidden=[4, 32])
    default_config = dict(num_layers=2, hidden=16, features_num=16)

    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SGConv(hidden, hidden))
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


class GCN(torch.nn.Module):
    config_desc = dict(num_layers=[1, 2], hidden=[4, 32])
    default_config = dict(num_layers=2, hidden=16, features_num=16)

    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2):
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


class Model(object):

    def __init__(self):
        """Constructor
        only `train_predict()` is measured for timing, put as much stuffs here as possible
        """

        self.device = torch.device('cuda:0' if torch.cuda.
                                   is_available() else 'cpu')

        # torch.nn.Module subclass with class attribute `config_desc` and `default_config`
        model_list = [GCN, SplineConv, SGCN]
        self._model_cls = model_list[1]
        # used by the scheduler for deciding when to stop each trial
        early_stopper = ConstantStopper(max_step=800)
        # schedulers conduct HPO
        self._scheduler = Scheduler(
            early_stopper, self._model_cls.config_desc,
            self._model_cls.default_config)
        # ensemble the promising models searched
        self._ensembler = GreedyStrategy(finetune=False)

    def generate_pyg_data(self, data):

        x = data['fea_table']
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
        else:
            x = x.drop('node_index', axis=1).to_numpy()

        x = torch.tensor(x, dtype=torch.float)

        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

        edge_weight = df['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype=torch.long)
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

        data.num_nodes = num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        return data

    def train(self, data, model, optimizer):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(
            model(data)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return {"logloss": loss}

    def valid(self, data, model):
        model.eval()
        # (jones.wz) TO DO: implement
        return {"logloss": .0, "accuracy": 1.0}
    
    def pred(self, model, data):
        model.eval()
        #data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred

    def train_predict(self, data, time_budget, n_class, schema):

        self._scheduler.setup_timer(time_budget)
        data = self.generate_pyg_data(data).to(self.device)
        # (jones.wz) TO DO: split data for HPO
        train_data, early_stop_valid_data, valid_data = data, None, data
        model = None
        while not self._scheduler.should_stop(FRAC_FOR_SEARCH):
            if model:
                # within a trial, just continue the training
                train_info = self.train(train_data, model, optimizer)
                if early_stop_valid_data:
                    early_stop_valid_info = self.valid(early_stop_valid_data, model)
                else:
                    early_stop_valid_info = None
                if self._scheduler.should_stop_trial(train_info, early_stop_valid_info):
                    valid_info = self.valid(valid_data, model)
                    self._scheduler.record(model, valid_info)
                    model, optimizer = None, None
            else:
                # trigger a new trial
                config = self._scheduler.get_next_config()
                if config:
                    config["features_num"] = data.x.size()[1]
                    config["num_class"] = n_class
                    model = self._model_cls(**config).to(self.device)
                    # (jones.wz) TO DO: refactor the hyperparam space to include \
                    # the optimization-related hyperparams
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=0.005, weight_decay=5e-4)
                else:
                    # exhaust the search space
                    break
        if model is not None:
            valid_info = self.valid(valid_data, model)
            self._scheduler.record(model, valid_info)
        
        model = self._ensembler.boost(
            data, self._scheduler.get_results(), self._model_cls).to(self.device)

        pred = self.pred(model, data)

        return pred.cpu().numpy().flatten()
