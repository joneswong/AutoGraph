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
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler
import subprocess
import os

logger=logger = logging.getLogger('code_submission')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_pyg_data(data, n_class, use_dim_reduction=True, use_feature_generation=False):
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
    

    ###   feature engineering  ###
    non_feature = False
    if use_dim_reduction:
        x, non_feature = dim_reduction(x)
    else:
        x = x.to_numpy()

    if x.shape[1] == 1:
        #x = x.to_numpy()
        #x = x.reshape(x.shape[0])
        #x = np.array(pd.get_dummies(x))
        non_feature = True
    #else:
        #x = x.drop('node_index', axis=1).to_numpy()
        #x = x.to_numpy()
        #ss = StandardScaler()
        #x = ss.fit_transform(x)
    
    #rw = np.load('e_v.npy')
    node_embed = run_STRAP(num_nodes, edge_index)
    x = np.concatenate([x,node_embed], axis=1)

    if use_feature_generation:
        added_features = feature_generation(x, y, n_class, edge_index, non_feature)
        x = np.concatenate([x]+added_features, axis=1)

    print(x.shape)
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

def run_STRAP(num_nodes, edges):
    start_time = time.time()
    file_path = os.path.dirname(__file__)

    data_dir = file_path + '/NR_Dataset'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    embed_dir = file_path + '/NR_EB'
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    edges = edges.numpy().transpose([1,0])

    #write edge file
    w_str = str(num_nodes)+'\n'
    num_edges = len(edges)
    if num_edges > 5e6:
        STRAP_epsilon = 5e-3
    else:
        STRAP_epsilon = 1e-4

    foo = lambda x: str(x[0])+' '+str(x[1]) if x[0] < x[1] else str(x[1])+' '+str(x[0])
    w_str += '\n'.join(map(foo , edges))
    #print(time.time() - start_time)
    with open(os.path.join(data_dir,'STRAP.txt'),'w') as f:
        f.write(w_str)

    #run_commands = "./code_submission/temp_STRAP_FRPCA_U STRAP ./code_submission/NR_Dataset/ ./code_submission/NR_EB/ 0.5 12 0.0001 24"
    run_commands = ' '.join([os.path.join(file_path,'STRAP_FRPCA_U'),
                    'STRAP',
                    data_dir+'/',
                    embed_dir+'/',
                    '0.5 12',
                    str(STRAP_epsilon),
                    '24 128'])
    #print(run_commands)
    rc, out = subprocess.getstatusoutput(run_commands)
    #print(rc, out)

    node_embed = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_u_U.csv'), header=None)
    if node_embed.isnull().values.any():
        node_embed.fillna(0.0)
        print('find nan in node_embed')

    time_cost = time.time() - start_time
    with open('time_cost','a') as f:
        f.write(str(time_cost)+'\n')
    print('time cost of runing STRAP: ',time_cost)
    
    return node_embed.to_numpy()

