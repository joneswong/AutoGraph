from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger('code_submission')

import random
import numpy as np
import pandas as pd
from pandas import Series
import torch
import time
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_add
import subprocess
import os

def _pca_processing(data, pca_threshold=0.75):
    if data.shape[1] == 0:
        return data
    pca=PCA(n_components=pca_threshold, svd_solver ='full')
    data = pca.fit_transform(data)
    return data

def _check_file_exist(file_path, flag_directed_graph):
    if flag_directed_graph and os.path.exists(os.path.join(file_path,'NR_EB/STRAP_strap_frpca_d_U.csv')) and os.path.exists(os.path.join(file_path,'NR_EB/STRAP_strap_frpca_d_V.csv')):
        return True
    elif (not flag_directed_graph) and os.path.exists(os.path.join(file_path,'NR_EB/STRAP_strap_frpca_u_V.csv')):
        return True
    else:
        return False

def check_monotony(x, tol=5):
    dx = np.diff(x[1:-1])
    return np.all(dx < tol) or np.all(dx > -tol)

def get_value_counts_with_moving_average(x, use_moving_average=False, n=3):
    x_dict = dict(Series(x).value_counts())
    x = [x_dict[i] for i in sorted(x_dict.keys())]

    if use_moving_average:
        ret = np.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        print(ret[n-1:]/n)
        return ret[n - 1:] / n
    else:
        return x

def check_continuous(x, tol=5):
    x = get_value_counts_with_moving_average(x)
    max_index = np.argmax(x)
    min_index = np.argmax(-np.array(x))
    if check_monotony(x, tol):
        return True
    elif check_monotony(x[:max_index+1], tol) and check_monotony(x[max_index:], tol):
        return True
    elif check_monotony(x[:min_index+1], tol) and check_monotony(x[min_index:], tol):
        return True
    else:
        return False
    
def normalize(x):
    norm_time = time.time()
    tol = min(int(1e-3*x.shape[1]), 5)
    normal_funs = ['l2', 'minmax', 'z-score']
    normal_fun = normal_funs[2]

    cont_feature_idx = [i for i in range(len(x)) if len(np.unique(x[i])) > 5 and check_continuous(x[i], tol)] 
    cate_feature_idx = [i for i in range(len(x)) if i not in cont_feature_idx]
    logger.info('# continous features: {}, # categorical features: {}'.format(len(cont_feature_idx), len(cate_feature_idx)))

    cate_feature = x[cate_feature_idx]
    cont_feature = x[cont_feature_idx]
    if len(cont_feature) > 0:
        if normal_fun == 'l2':
            norm = list(map(lambda y: np.linalg.norm(y,keepdims=True), cont_feature))
            cont_feature = cont_feature/np.array(norm, dtype=np.float32)
        elif normal_fun == 'min-max':
            min_value = np.min(cont_feature, 1, keepdims=True)
            max_value = np.max(cont_feature, 1, keepdims=True)
            cont_feature = (cont_feature-min_value) / (max_value-min_value)
        elif normal_fun == 'z-score':
            mean_value = np.mean(cont_feature, 1, keepdims=True)
            std_value = np.std(cont_feature, 1, keepdims=True)
            cont_feature = (cont_feature-mean_value) / std_value

    normalized_features = np.concatenate([cate_feature, cont_feature], axis=0)

    logger.info('normlization time cost: {}'.format(time.time()-norm_time))
    return normalized_features.transpose(1,0)

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

def get_node_degree(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    in_deg = scatter_add(edge_weight, col, dim_size=num_nodes).numpy()
    out_deg = scatter_add(edge_weight, row, dim_size=num_nodes).numpy()
    degree = np.concatenate([np.expand_dims(in_deg,-1), np.expand_dims(out_deg,-1)], axis=-1)
    return degree

def get_node_degree_binary(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    in_deg = scatter_add(edge_weight, col, dim_size=num_nodes)
    in_deg_binary = torch.ones_like(in_deg)
    in_deg_binary[torch.nonzero(in_deg).reshape(-1)] = 0.0
    in_deg_binary = in_deg_binary.numpy()

    out_deg = scatter_add(edge_weight, row, dim_size=num_nodes)
    out_deg_binary = torch.ones_like(out_deg)
    out_deg_binary[torch.nonzero(out_deg).reshape(-1)] = 0.0
    out_deg_binary = out_deg_binary.numpy()
    degree_binary = np.concatenate([np.expand_dims(in_deg_binary,-1), np.expand_dims(out_deg_binary,-1)], axis=-1)
    return degree_binary

def _foo_directed(x):
    return str(x[0])+' '+str(x[1])

def _foo_undirected(x):
    return str(x[0])+' '+str(x[1]) if x[0] < x[1] else str(x[1])+' '+str(x[0])

def run_STRAP(num_nodes, edges, weights, flag_directed_graph, flag_none_feature, time_budget, epsilon=1e6, dims=128):
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
    if num_edges < epsilon: 
        STRAP_epsilon = 1e-4
        timeout = int(0.2*time_budget)
    elif num_edges < 10*epsilon and flag_none_feature:
        STRAP_epsilon = 5e-4
        timeout = int(0.3*time_budget)
    else:
        STRAP_epsilon = 1e-3
        timeout = int(0.3*time_budget)

    if flag_directed_graph:
        _tmp = map(_foo_directed, edges)
    else:
        _tmp = map(_foo_undirected, edges)
    write_str =str(num_nodes)+'\n' + '\n'.join(_tmp)

    with open(os.path.join(data_dir,'STRAP.txt'),'w') as f:
        f.write(write_str)

    #run_commands = "./code_submission/temp_STRAP_FRPCA_U STRAP ./code_submission/NR_Dataset/ ./code_submission/NR_EB/ 0.5 12 0.0001 24"
    STRAP_file = 'STRAP_FRPCA_D' if flag_directed_graph else 'STRAP_FRPCA_U'
    try:
        run_commands = ' '.join(['chmod','u+x',os.path.join(file_path,STRAP_file)])
        cmd_return = subprocess.run(run_commands, shell=True, timeout=1)
        #logger.info('chomod commands return: {}'.format(proc.returncode))

        run_commands = ' '.join([os.path.join(file_path,STRAP_file),
                        'STRAP', data_dir+'/', embed_dir+'/',
                        '0.5 12', str(STRAP_epsilon), '8', str(dims)])
        cmd_return = subprocess.run(run_commands, shell=True, timeout=timeout)
        #logger.info('chomod commands return: {}'.format(proc.returncode))
    except subprocess.TimeoutExpired as timeout_msg:
        flag_error = True
        logger.info('STRAP timeout! error msg: {}'.format(timeout_msg))
    except Exception as err_msg:
        flag_error = True
        logger.info('STRAP failed with other errors! error msg: {}'.format(err_msg))
    else:
        flag_error = False
    finally:
        if not flag_error and _check_file_exist(file_path, flag_directed_graph):
        
            if flag_directed_graph:
                node_embed_u = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_d_U.csv'), header=None)
                if node_embed_u.isnull().values.any():
                    node_embed_u.fillna(0.0)
                    logger.info('find nan in node_embed_U')
                node_embed_v = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_d_V.csv'), header=None)
                if node_embed_v.isnull().values.any():
                    node_embed_v.fillna(0.0)
                    logger.warn('find nan in node_embed_V')
                node_embed = np.concatenate([node_embed_u, node_embed_v], axis=1)
            else:
                node_embed = pd.read_csv(os.path.join(file_path, 'NR_EB/STRAP_strap_frpca_u_U.csv'), header=None)
                if node_embed.isnull().values.any():
                    node_embed_u.fillna(0.0)
                    logger.warn('find nan in node_embed_U')
        else:
            logger.warn('Error: no such file!')
            node_embed = []
    
    return flag_error, node_embed
    
def dim_reduction(x, use_normalizer=False):
    #remove uninformative col

    drop_col = [col for col in x.columns if x[col].var() == 0]
    #all the features are uninformative except node_index
    flag_none_feature = (len(drop_col) == len(x.columns)-1)
    x = x.drop(drop_col,axis=1).to_numpy()
    
    if not flag_none_feature and use_normalizer:
        x = np.concatenate([x[:,0:1], normalize(x[:,1:].transpose(1,0))], axis=1)

    return x, flag_none_feature

def feature_generation(x, y, n_class, edges, weights, flag_none_feature, flag_directed_graph, time_budget, use_label_distribution=False, use_node_degree=False, use_node_degree_binary=False, use_node_embed=True):

    added_features = []
    start_time = time.time()
    num_nodes = x.shape[0]

    if flag_none_feature and use_label_distribution:
        label_distribution = get_neighbor_label_distribution(edges, y, n_class) 
        added_features.append(label_distribution)
        logger.info('neighbor_label_distribution time cost: {}'.format(time.time() - start_time))

    if use_node_degree:
        node_degree = get_node_degree(edges, weights, num_nodes)
        added_features.append(node_degree)
        logger.info('degree time_cost: '.format(time.time() - start_time))

    if use_node_degree_binary:
        node_degree_binary = get_node_degree_binary(edges, weights, num_nodes)
        added_features.append(node_degree_binary)
        logger.info('degree_binary time_cost: {}'.format(time.time() - start_time))

    if use_node_embed:
        flag_error, node_embed = run_STRAP(num_nodes, edges, weights, flag_directed_graph, flag_none_feature, time_budget)
        if not flag_error:
            added_features.append(node_embed)
        logger.info('node_embed time cost: {}'.format(time.time() - start_time))

    return added_features
