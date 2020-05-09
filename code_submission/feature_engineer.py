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
    for edge in edges:
        src_idx = edge[0]
        dst_idx = edge[1]
        distribution[src_idx][y[dst_idx]] += 1.0
        distribution[dst_idx][y[src_idx]] += 1.0

    norm_matrix = np.sum(distribution[:,:-1], axis=1, keepdims=True) + EPSILON
    distribution = distribution[:,:-1] / norm_matrix
    
    return distribution

def get_node_degree(nodes, num_nodes):
    node_degree = degree(nodes, num_nodes)
    return np.expand_dims(node_degree, axis=-1)
    

def dim_reduction(x):
    #remove uninformative col
    index_col = x['node_index']
    drop_col = [col for col in x.columns if x[col].var() == 0] + ['node_index']
    if len(drop_col) == len(x.columns):
        x = np.expand_dims(index_col.to_numpy(),axis=-1)
        non_feature = True
    else:
        x = x.drop(drop_col,axis=1).to_numpy()
        non_feature = False
    
    return x, non_feature

def feature_generation(x, y, n_class, edges, non_feature, use_label_distribution=True, use_node_degree=True):
    added_features = []
    
    #start_time = time.time()
    if non_feature and use_label_distribution:
        label_distribution = get_neighbor_label_distribution(edges, y, n_class) 
        added_features.append(label_distribution)
        #print(label_distribution.shape)

    #print('label_dis', time.time() - start_time)

    if non_feature and use_node_degree:
        node_degree = get_node_degree(edges[0], x.shape[0])
        added_features.append(node_degree)
        #print(node_degree.shape)

    #print('degree_time', time.time() - start_time)
    return added_features
