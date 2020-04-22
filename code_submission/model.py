"""the simple baseline for autograph"""

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_geometric.data import Data

import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)


class GCN(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=16,  features_num=16, num_class=2):
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


class Model:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    def train(self, data):
        model = GCN(features_num=data.x.size()[1], num_class=int(max(data.y)) + 1)
        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        min_loss = float('inf')
        for epoch in range(1,800):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        return model

    def pred(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            pred = model(data)[data.test_mask].max(1)[1]
        return pred

    def train_predict(self, data, time_budget,n_class,schema):


        data = self.generate_pyg_data(data)
        model = self.train(data)
        pred = self.pred(model, data)

        return pred.cpu().numpy().flatten()
