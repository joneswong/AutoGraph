from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import pandas as pd
import torch
import time
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler
import subprocess
import os


def _pca_processing(data, pca_threshold=0.75):
    if data.shape[1] == 0:
        return data
    pca=PCA(n_components=pca_threshold, svd_solver ='full')
    data = pca.fit_transform(data)
    return data

def get_neighbor_label_distribution(edges, y, n_class):
    EPSILON = 1e-8
    num_nodes = len(y)
    distribution= np.zeros([num_nodes, n_class+1], dtype=np.float32)
    edges = edges.numpy().transpose([1,0])
    for edge in edges:
        src_idx = edge[0]
        dst_idx = edge[1]
        distribution[src_idx][y[dst_idx]] += 1.0

    # the last dimension is 'unknow' (the labels of test nodes)
    norm_matrix = np.sum(distribution[:,:-1], axis=1, keepdims=True) + EPSILON
    distribution = distribution[:,:-1] / norm_matrix
    
    return distribution

def get_node_degree(nodes, num_nodes):
    node_degree = degree(nodes, num_nodes)
    return np.expand_dims(node_degree, axis=-1)

def run_STRAP(num_nodes, edges, flag_directed_graph, epsilon=1e6, dims=128):
    file_path = os.path.dirname(__file__)
    data_dir = file_path + '/NR_Dataset'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    embed_dir = file_path + '/NR_EB'
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    edges = edges.numpy().transpose([1,0])

    #write edge file
    num_edges = len(edges)
    if num_edges > epsilon:
        STRAP_epsilon = 5e-3
    else:
        STRAP_epsilon = 1e-4

    if flag_directed_graph:
        foo = lambda x: str(x[0])+' '+str(x[1])
    else:
        foo = lambda x: str(x[0])+' '+str(x[1]) if x[0] < x[1] else str(x[1])+' '+str(x[0])
    write_str =str(num_nodes)+'\n' + '\n'.join(map(foo , edges))

    with open(os.path.join(data_dir,'STRAP.txt'),'w') as f:
        f.write(write_str)

    #run_commands = "./code_submission/temp_STRAP_FRPCA_U STRAP ./code_submission/NR_Dataset/ ./code_submission/NR_EB/ 0.5 12 0.0001 24"
    STRAP_file = 'STRAP_FRPCA_D' if flag_directed_graph else 'STRAP_FRPCA_U'
    run_commands = ' '.join(['chmod','u+x',os.path.join(file_path,STRAP_file)])
    rc, out = subprocess.getstatusoutput(run_commands)
    print('chomod commands return: ', rc, out)

    run_commands = ' '.join([os.path.join(file_path,STRAP_file),
                    'STRAP', data_dir+'/', embed_dir+'/',
                    '0.5 12', str(STRAP_epsilon), '8', str(dims)])
    rc, out = subprocess.getstatusoutput(run_commands)
    print('STRAP_commands return: ', rc, out)

    if flag_directed_graph:
        node_embed_u = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_d_U.csv'), header=None)
        if node_embed_u.isnull().values.any():
            node_embed_u.fillna(0.0)
            print('find nan in node_embed_U')
        node_embed_v = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_d_V.csv'), header=None)
        if node_embed_v.isnull().values.any():
            node_embed_v.fillna(0.0)
            print('find nan in node_embed_V')
        node_embed = np.concatenate([node_embed_u, node_embed_v], axis=1)
    else:
        node_embed = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_u_U.csv'), header=None)
        if node_embed.isnull().values.any():
            node_embed_u.fillna(0.0)
            print('find nan in node_embed_U')
    
    return node_embed
    
def dim_reduction(x):
    #remove uninformative col

    drop_col = [col for col in x.columns if x[col].var() == 0]
    #all the features are uninformative except node_index
    flag_none_feature = (len(drop_col) == len(x.columns)-1)
    x = x.drop(drop_col,axis=1).to_numpy()

    return x, flag_none_feature

def feature_generation(x, y, n_class, edges, flag_none_feature, flag_directed_graph, 
        use_label_distribution=False, use_node_degree=False, use_node_embed=True):

    added_features = []
    start_time = time.time()
    num_nodes = x.shape[0]

    if flag_none_feature and use_label_distribution:
        label_distribution = get_neighbor_label_distribution(edges, y, n_class) 
        added_features.append(label_distribution)
        print('neighbor_label_distribution time cost: ', time.time() - start_time)

    if use_node_degree:
        node_degree = get_node_degree(edges[0], num_nodes)
        added_features.append(node_degree)
        print('degree time_cost: ', time.time() - start_time)

    if use_node_embed:
        node_embed = run_STRAP(num_nodes, edges, flag_directed_graph)
        added_features.append(node_embed)
        print('node_embed time cost: ', time.time() - start_time)

    return added_features
