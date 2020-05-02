from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import pandas as pd
import torch
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


def dim_reduction(x, sparse_threshold=0.9, pca_threshold=0.75):
    #remove uninformative col
    index_col = x['node_index']
    drop_col = [col for col in x.columns if x[col].var() == 0] + ['node_index']
    x = x.drop(drop_col,axis=1)
    # pca
    if x.shape[1] != 0:
        sparse_col = [col for col in x.columns if x[col].value_counts()[0] > sparse_threshold*x.shape[0]]
        dense_col = [col for col in x.columns if col not in sparse_col]
        sparse_feature = x[sparse_col]
        dense_feature = x[dense_col]
        pre_sparse_feature = _pca_processing(sparse_feature, pca_threshold)
        pre_dense_feature = dense_feature.to_numpy()
        pre_x = np.concatenate([pre_sparse_feature,pre_dense_feature],axis=1)
    else:
        pre_x = np.expand_dims(index_col.to_numpy(),axis=-1)
    return pre_x


def feature_generation(x, edge_index):
    # new_feature: node degree
    row, col = edge_index
    node_degree = degree(row, x.shape[0])
    return np.expand_dims(node_degree, axis=-1)