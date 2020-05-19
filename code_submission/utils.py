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
from torch_geometric.utils import degree, is_undirected
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger('code_submission')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pyg_data(data, n_class, time_budget, use_dim_reduction=True, use_feature_generation=True):
    x = data['fea_table']

    df = data['edge_file']
    edge_index = df[['src_idx', 'dst_idx']].to_numpy()
    edge_index = sorted(edge_index, key=lambda d: d[0])
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)

    edge_weight = df['edge_weight'].to_numpy()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    num_nodes = x.shape[0]
    y = n_class * torch.ones(num_nodes, dtype=torch.long)
    inds = data['train_label'][['node_index']].to_numpy()
    train_y = data['train_label'][['label']].to_numpy()
    y[inds] = torch.tensor(train_y, dtype=torch.long)

    train_indices = data['train_indices']
    test_indices = data['test_indices']
    

    flag_directed_graph = not is_undirected(edge_index)

    ###   feature engineering  ###
    flag_none_feature = False
    if use_dim_reduction:
        x, flag_none_feature = dim_reduction(x)
    else:
        x = x.to_numpy()
        flag_none_feature = (x.shape[1] == 1)
    
    if use_feature_generation:
        added_features = feature_generation(x, y, n_class, edge_index, edge_weight,  flag_none_feature, flag_directed_graph, time_budget)
        x = np.concatenate([x]+added_features, axis=1)

    if x.shape[1] != 1:
        #remove raw node_index 
        x = x[:,1:]

    logger.info('x.shape after feature engineering: {}'.format(x.shape))
    x = torch.tensor(x, dtype=torch.float)

    non_zero_index = torch.nonzero(edge_weight).reshape(-1)
    edge_weight = edge_weight[non_zero_index]
    edge_index = edge_index[:,non_zero_index]
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


def get_label_weights(train_label, n_class):
    """
        return label balanced weights according to the label distribution
    """
    unique, counts = np.unique(train_label, return_counts=True)
    if not len(counts) == n_class:
        raise ValueError("Your train_label has different label size to the meta_n_class")
    inversed_counts = 1.0 / counts
    # is_major = (counts > 100).astype(float)
    # inversed_counts = 1.0 / counts * is_major + 100.0 * (1.0 - is_major)
    normalize_factor = inversed_counts.sum()
    inversed_counts = inversed_counts / normalize_factor

    T = 1.0
    inversed_counts = np.power(inversed_counts, T)

    # return [1.0 / n_class] * n_class  # the same weights for all label class
    return inversed_counts


def generate_pyg_data_without_transform(data):
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


def hyperparam_space_tostr(hyperparam_space):
    hyperparam_space_str = '\n'
    for k, v in hyperparam_space.items():
        hyperparam_space_str = hyperparam_space_str + "%-15s: %s\n" % (k, v)
    return hyperparam_space_str


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

def calculate_config_dist(tpa, tpb):
    """Trivially calculate the distance of two configs"""

    ca, cb = tpa[0], tpb[0]
    num_diff_field = 0
    for k in ca:
        if ca[k] != cb[k]:
            num_diff_field += 1
    return num_diff_field


def divide_data_label_wise(data, split_rates, device, n_class, train_y):
    # divide training data into several partitions, according to the label distribution

    train_indices_label_wise = dict()
    for i in range(n_class):
        train_indices_label_wise[i] = []

    for i, train_idx in enumerate(data.train_indices):
        data_i_y = int(train_y[i])
        train_indices_label_wise[data_i_y].append(train_idx)

    all_indices = [[] for _ in range(len(split_rates))]
    for data_of_y_i in train_indices_label_wise.values():
        indices = np.array(data_of_y_i)
        np.random.shuffle(indices)

        split_thred = []
        accumulated_rate = 0
        for r in split_rates:
            accumulated_rate += r
            split_thred.append(int(len(indices)*accumulated_rate/np.sum(split_rates)))

        prev = 0
        for i, end in enumerate(split_thred):
            part_indices = indices[prev:end] if i < len(split_thred)-1 else indices[prev:]
            prev = end
            # y_i_indices.append(part_indices)
            all_indices[i].extend(part_indices)

    masks = list()
    for i in range(len(all_indices)):
        part_masks = torch.zeros(data.num_nodes, dtype=torch.bool)
        part_masks[all_indices[i]] = 1
        masks.append(part_masks.to(device))
    return tuple(masks)
