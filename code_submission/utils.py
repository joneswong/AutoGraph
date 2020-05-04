from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from feature_engineer import dim_reduction, feature_generation

logger=logger = logging.getLogger('code_submission')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pyg_data(data, use_dim_reduction=True, use_feature_generation=True, sparse_threshold=0.9, pca_threshold=0.75):

    x = data['fea_table']

    df = data['edge_file']
    edge_index = df[['src_idx', 'dst_idx']].to_numpy()
    edge_index = sorted(edge_index, key=lambda d: d[0])
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)


    edge_weight = df['edge_weight'].to_numpy()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    num_nodes = x.shape[0]
    y = torch.zeros(num_nodes, dtype=torch.long)
    inds = data['train_label'][['node_index']].to_numpy()
    train_y = data['train_label'][['label']].to_numpy()
    y[inds] = torch.tensor(train_y, dtype=torch.long)

    train_indices = data['train_indices']
    test_indices = data['test_indices']

    ###   feature engineering  ###
    if use_dim_reduction:
        x = dim_reduction(x, sparse_threshold, pca_threshold)
    else:
        if x.shape[1] == 1:
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
        else:
            x = x.drop('node_index', axis=1).to_numpy()

    if x.shape[1] == 1 and use_feature_generation:
        added_feature = feature_generation(x, edge_index)
        x = np.concatenate([x, added_feature], axis=1)
    
    x = torch.tensor(x, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)

    data.num_nodes = num_nodes
    data.train_indices = train_indices
    data.test_indices = test_indices

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
    # return -valid_info['logloss']+0.1*valid_info['accuracy']
    return valid_info['accuracy']


def divide_data(data, split_rates, device):
    # divide training data into several partitions
    indices = np.array(data.train_indices)
    np.random.shuffle(indices)

    split_thred = []
    accumulated_rate = 0
    for r in split_rates:
        accumulated_rate += r
        split_thred.append(int(len(indices)*accumulated_rate/np.sum(split_rates)))

    all_indices = list()
    prev = 0
    for i, end in enumerate(split_thred):
        part_indices = indices[prev:end] if i < len(split_thred)-1 else indices[prev:]
        prev = end
        all_indices.append(part_indices)

    masks = list()
    for i in range(len(all_indices)):
        part_masks = torch.zeros(data.num_nodes, dtype=torch.bool)
        part_masks[all_indices[i]] = 1
        masks.append(part_masks.to(device))
    return tuple(masks)
