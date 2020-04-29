from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pyg_data(data):

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


def get_performance(valid_info):
    # the larger, the better
    # naive implementation
    return -valid_info['logloss']+0.1*valid_info['accuracy']
