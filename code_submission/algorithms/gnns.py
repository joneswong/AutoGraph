from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
from torch.nn import Linear, functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, SGConv, SplineConv, APPNP
from .gnn_tricks import GraphSizeNorm
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch.nn as nn

from torch_geometric.utils.dropout import dropout_adj


def adaptive_message_func(edges):
    """
    send data for computing metrics and update.
    """
    return {'feat': edges.src['h'], 'logits': edges.src['logits']}


def adaptive_attn_message_func(edges):
    return {'feat': edges.src['ft'] * edges.data['a'], 'logits': edges.src['logits'], 'a': edges.data['a']}


def adaptive_attn_reduce_func(nodes):
    # (n_nodes, n_edges, n_classes)
    _, pred = torch.max(nodes.mailbox['logits'], dim=2)
    _, center_pred = torch.max(nodes.data['logits'], dim=1)
    n_degree = nodes.data['degree']
    # case 1
    # ratio of common predictions
    a = nodes.mailbox['a'].squeeze(3)  # (n_node, n_neighbor, n_head, 1)
    n_head = a.size(2)
    idxs = torch.eq(pred, center_pred.unsqueeze(1)).unsqueeze(2).expand_as(a)
    f1 = torch.div(torch.sum(a * idxs, dim=1), n_degree.unsqueeze(1))  # (n_node, n_head)
    f1 = f1.detach()
    # case 2
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    # (n_unique)
    cnts_p = torch.zeros((pred.size(0), n_head, uniq.size(0),)).cuda()
    for i, val in enumerate(uniq):
        idxs = torch.eq(pred, val).unsqueeze(2).expand_as(a)
        tmp = torch.div(torch.sum(a * idxs, dim=1), n_degree.unsqueeze(1))  # (n_nodes, n_head)
        cnts_p[:, :, i] = tmp
    cnts_p = torch.clamp(cnts_p, min=1e-5)
    f2 = (-1) * torch.sum(cnts_p * torch.log(cnts_p), dim=2)
    f2 = f2.detach()
    neighbor_agg = torch.sum(nodes.mailbox['feat'], dim=1)  # (n_node, n_head, n_feat)
    return {
        'f1': f1,
        'f2': f2,
        'agg': neighbor_agg,
    }


class GatedLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, lidx=1):
        super(GatedLayer, self).__init__()
        self.weight_neighbors = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.tau_1 = nn.Parameter(torch.zeros((1,)))
        self.tau_2 = nn.Parameter(torch.zeros((1,)))

        self.ln_1 = nn.LayerNorm((g.number_of_nodes()), elementwise_affine=False)
        self.ln_2 = nn.LayerNorm((g.number_of_nodes()), elementwise_affine=False)

        self.reset_parameters(lidx)

    def reset_parameters(self, lidx, how='layerwise'):
        # initialize params
        if how == 'normal':
            nn.init.normal_(self.tau_1)
            nn.init.normal_(self.tau_2)
        else:
            nn.init.constant_(self.tau_1, 1 / (lidx + 1))
            nn.init.constant_(self.tau_2, 1 / (lidx + 1))
        return

    def forward(self, g, h, logits, old_z, shared_tau=True, tau_1=None, tau_2=None):
        # TODO: training mode different from test mode
        # operates on a node
        g = g.local_var()
        if self.dropout:
            h = self.dropout(h)
        # g.ndata['h'] = h * g.ndata['norm'] # normalization by src
        g.ndata['h'] = h
        g.ndata['logits'] = logits

        g.update_all(message_func=fn.copy_u('logits', 'logits'), reduce_func=adaptive_reduce_func)
        f1 = g.ndata.pop('f1')
        f2 = g.ndata.pop('f2')
        norm_f1 = self.ln_1(f1)
        norm_f2 = self.ln_2(f2)
        if shared_tau:
            z = F.sigmoid((-1) * (norm_f1 - tau_1)) * F.sigmoid((-1) * (norm_f2 - tau_2))
        else:
            # tau for each layer
            z = F.sigmoid((-1) * (norm_f1 - self.tau_1)) * F.sigmoid((-1) * (norm_f2 - self.tau_2))

        gate = torch.min(old_z, z)
        g.update_all(message_func=fn.copy_u('h', 'feat'), reduce_func=fn.sum(msg='feat', out='agg'))

        agg = g.ndata.pop('agg')

        # normagg = self.weight_neighbors(agg)*g.ndata['norm']
        normagg = agg * g.ndata['norm']  # normalization by tgt degree

        if self.activation:
            normagg = self.activation(normagg)
        new_h = h + gate.unsqueeze(1) * normagg
        return new_h, z


class DirectedGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(DirectedGCNConv, self).__init__(in_channels, out_channels, improved, cached, bias, **kwargs)
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, int(out_channels / 2)))  # 2 is due to hidden concat
        torch.nn.init.xavier_uniform_(self.weight, gain=1)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col], \
               deg_inv_sqrt[col] * edge_weight * deg_inv_sqrt[row]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm_in, norm_out = self.norm(edge_index, x.size(0), edge_weight,
                                                      self.improved, x.dtype)
            self.cached_result = edge_index, norm_in, norm_out
        edge_index, norm_in, norm_out = self.cached_result
        return self.propagate(edge_index, x=x, norm_in=norm_in, norm_out=norm_out)

    def message(self, x_j, norm_in, norm_out):
        in_f = norm_in.view(-1, 1) * x_j
        out_f = norm_out.view(-1, 1) * x_j
        return torch.cat((in_f, out_f), 1)


class SplineGCN(torch.nn.Module):

    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, droprate=0.5, dim=1, kernel_size=2,
                 edge_droprate=0.0, fea_norm="no_norm"):
        super(SplineGCN, self).__init__()
        self.droprate = droprate
        self.edge_droprate = edge_droprate
        if fea_norm == "no_norm":
            self.fea_norm_layer = None
        elif fea_norm == "graph_size_norm":
            self.fea_norm_layer = GraphSizeNorm()
        else:
            raise ValueError("your fea_norm is un-defined: %s") % fea_norm
        # todo (daoyuan) add more weight init method

        self.convs = torch.nn.ModuleList()
        self.convs.append(SplineConv(features_num, hidden, dim, kernel_size))
        for i in range(num_layers - 2):
            self.convs.append(SplineConv(hidden, hidden, dim, kernel_size))
        self.convs.append(SplineConv(hidden, num_class, dim, kernel_size))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for i, conv in enumerate(self.convs):
            # todo (daoyuan) add layer_norm
            x = x if self.fea_norm_layer is None else self.fea_norm_layer(x)
            x = F.dropout(x, p=self.droprate, training=self.training)
            if i == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = F.elu(conv(x, edge_index, edge_weight))
        # return F.log_softmax(x, dim=-1)
        # due to focal loss: return the logits, put the log_softmax operation into the GNNAlgo
        return x

    def __repr__(self):
        return self.__class__.__name__


class SplineGCN_APPNP(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, droprate=0.5, dim=1, kernel_size=2,
                 edge_droprate=0.0, fea_norm="no_norm", K=20, alpha=0.5):
        super(SplineGCN, self).__init__()
        self.droprate = droprate
        self.edge_droprate = edge_droprate
        if fea_norm == "no_norm":
            self.fea_norm_layer = None
        elif fea_norm == "graph_size_norm":
            self.fea_norm_layer = GraphSizeNorm()
        else:
            raise ValueError("your fea_norm is un-defined: %s") % fea_norm

        self.convs = torch.nn.ModuleList()
        self.convs.append(SplineConv(features_num, hidden, dim, kernel_size))
        for i in range(num_layers - 2):
            self.convs.append(SplineConv(hidden, hidden, dim, kernel_size))
        self.convs.append(SplineConv(hidden, num_class, dim, kernel_size))

        self.appnp = APPNP(K, alpha)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for conv in self.convs:
            x = x if self.fea_norm_layer is None else self.fea_norm_layer(x)
            x = F.dropout(x, p=self.droprate, training=self.training)
            x = F.elu(conv(x, edge_index, edge_weight))
        x = self.appnp(x)
        # return F.log_softmax(x, dim=-1)
        # due to focal loss: return the logits, put the log_softmax operation into the GNNAlgo
        return x

    def __repr__(self):
        return self.__class__.__name__


class SGCN(torch.nn.Module):
    def __init__(self, num_layers=2, hidden=16, features_num=16, num_class=2, hidden_droprate=0.5, edge_droprate=0.0):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SGConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.first_lin = Linear(features_num, hidden)
        self.hidden_droprate = hidden_droprate
        self.edge_droprate = edge_droprate

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        # due to focal loss: return the logits, put the log_softmax operation into the GNNAlgo
        return x

    def __repr__(self):
        return self.__class__.__name__
