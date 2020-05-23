from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils.dropout import dropout_adj
from sklearn.metrics import accuracy_score
from .gnns import DirectedGCNConv

from spaces import Categoric, Numeric

logger = logging.getLogger('code_submission')


# todo (daoyuan) change the GCNConv to DirectedGCNConv
class GCN(torch.nn.Module):
    def __init__(self,
                 num_class,
                 features_num,
                 num_layers=2,
                 hidden=16,
                 hidden_droprate=0.5, edge_droprate=0.0, res_type=0.0, directed=False):

        super(GCN, self).__init__()
        self.first_lin = Linear(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        if directed:
            self.convs.append(DirectedGCNConv(hidden, hidden * 2))
            hidden = hidden * 2
        else:
            self.convs.append(GCNConv(hidden, hidden))
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.hidden_droprate = hidden_droprate
        self.edge_droprate = edge_droprate
        self.res_type = res_type
        self.directed = directed

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        if self.res_type == 0.0:
            x = F.relu(self.first_lin(x))
            x = F.dropout(x, p=self.hidden_droprate, training=self.training)
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.hidden_droprate, training=self.training)
            x = self.lin2(x)
        else:
            x = F.relu(self.first_lin(x))
            x = F.dropout(x, p=self.hidden_droprate, training=self.training)
            x_list = [] if self.directed else [x]
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
                if self.res_type == 3.0 and len(x_list) != 0:
                    x = x + x_list[0]
                x_list.append(x)
            if self.res_type == 1.0:
                x = x + x_list[0]
            elif self.res_type == 2.0:
                x = torch.sum(torch.stack(x_list, 0), 0)
            x = F.dropout(x, p=self.hidden_droprate, training=self.training)
            x = self.lin2(x)

        # return F.log_softmax(x, dim=-1)
        # due to focal loss: return the logits, put the log_softmax operation into the GNNAlgo
        return x

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def bottleneck_tensor_size(cls,
                               num_nodes,
                               num_edges,
                               n_class,
                               features_num,
                               num_layers=2,
                               hidden=16,
                               hidden_droprate=0.5,
                               edge_droprate=0.0):
        """estimate the (gpu/cpu) memory consumption (in Byte) of the largest allocation"""
        # each float/int occupies 4 bytes
        return 4 * (num_nodes+num_edges) * hidden


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, device=torch.device('cpu'), reduction="mean", is_minority=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.reduction = reduction
        self.is_minority = torch.tensor(is_minority, dtype=torch.float32, device=device).view(1, -1) if is_minority is not None else None
        print(self.is_minority)
        self._EPSILON = 1e-7
        self.alpha = alpha
        if isinstance(alpha, list) or isinstance(alpha, np.ndarray):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=device).view(1, -1)
        else:
            raise ValueError("Your alpha should be a weight list with shape 1 * n, n is the label number.")

    def forward(self, input, target, T=1.0):
        """
        :param input: shape N * C
        :param target: shape N, categorical_target format
        :return: loss
        """
        if not len(target.shape) == 1:
            raise ValueError("Your target shape should be categorical format. Got: {}.".format(target.shape))
        if not len(input.shape) == 2:
            raise ValueError("Your input shape should be N * C. Got: {}.".format(input.shape))

        N, C = input.shape
        # transform categorical_target into onehot_target
        categorical_y = target.view(-1, 1)
        one_hot_target = torch.zeros([N, C], dtype=torch.float32, device=categorical_y.device)
        one_hot_target.scatter_(1, categorical_y, 1)

        preds = F.softmax(input)
        pt = preds * one_hot_target + (1.0 - preds) * (1.0 - one_hot_target)

        if self.is_minority is not None:
            # soft implementation
            # loss = -torch.sum(torch.pow(1 - pt, self.gamma).detach() * torch.log(pt + 1e-10), dim=1)

            # hard implementation
            loss = -torch.sum(one_hot_target * torch.pow(1 - pt, self.gamma).detach() * torch.log(pt + 1e-10), dim=1)

            weight = torch.sum((self.is_minority * T + (1.0 - self.is_minority)) * one_hot_target, dim=1)
            loss = weight * loss
        else:
            # soft implementation
            loss = -torch.sum(torch.pow(1 - pt, self.gamma).detach() * torch.log(pt + 1e-10), dim=1)

            # hard implementation
            # loss = -torch.sum(one_hot_target * torch.pow(1 - pt, self.gamma).detach() * torch.log(pt + 1e-10), dim=1)

            weight = torch.sum(one_hot_target * self.alpha, dim=1)
            loss = weight * loss

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))
        return loss


class GNNAlgo(object):
    """Encapsulate the torch.Module and the train&valid&test routines"""

    # hyperparam_space = dict(
    #     num_layers=Categoric(list(range(2,5)), None, 2),
    #     hidden=Categoric([16, 32, 64, 128], None, 16),
    #     dropout_rate=Numeric((), np.float32, 0.2, 0.6, 0.5),
    #     lr=Numeric((), np.float32, 1e-4, 1e-2, 5e-3),
    #     weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4))

    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        dropout_rate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4))

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config,
                 non_hpo_config):

        self._num_class = num_class
        self._features_num = features_num
        self._device = device
        self._num_class = num_class
        self.model = GCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16), config.get("hidden_droprate", 0.5)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self.loss_type = config.get("loss_type", "focal_loss")
        self.fl_loss = FocalLoss(config.get("gamma", 2), non_hpo_config.get("label_alpha", []), device)

    def train(self, data, data_mask, T = 1.0):
        self.model.train()
        self._optimizer.zero_grad()
        if self.loss_type == "focal_loss":
            loss = self.fl_loss(self.model(data)[data_mask], data.y[data_mask], T)
        elif self.loss_type == "ce_loss":
            loss = F.cross_entropy(self.model(data)[data_mask], data.y[data_mask])
        else:
            raise ValueError("You give the wrong loss type. Got {}.".format(self.loss_type))
        loss.backward()
        self._optimizer.step()
        # We may need values other than loss for making better decisions on
        # when to stop the training course
        return {"loss": loss.item()}

    def valid(self, data, data_mask):
        self.model.eval()
        with torch.no_grad():
            validation_output = self.model(data)[data_mask]
            validation_pre = validation_output.max(1)[1]
            validation_truth = data.y[data_mask]
            if self.loss_type == "focal_loss":
                loss = self.fl_loss(self.model(data)[data_mask], data.y[data_mask])
            elif self.loss_type == "ce_loss":
                loss = F.cross_entropy(validation_output, validation_truth)
            else:
                raise ValueError("You give the wrong loss type. Got {}.".format(self.loss_type))

        cpu = torch.device('cpu')
        accuracy = accuracy_score(validation_truth.to(cpu), validation_pre.to(cpu))
        return {"loss": loss.item(), "accuracy": accuracy}
    
    def pred(self, data, make_decision=True):
        self.model.eval()
        with torch.no_grad():
            if make_decision:
                pred = self.model(data)[data.test_mask].max(1)[1]
            else:
                pred = self.model(data)[data.test_mask]
        # WARN: from the original master (no focal loss version) to dev_daoyuan version,
        # the pred is changed from log_softmax(x) into logits x, due to adapt to focal loss
        return pred

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, filter_out=False, strict=True):
        prev_state_dict = torch.load(path, map_location=self._device)
        if filter_out:
            prev_state_dict = {k: v for k, v in prev_state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(prev_state_dict, strict=strict)


class GCNAlgo(GNNAlgo):

    hyperparam_space = dict(
        num_layers=Categoric(list(range(1, 4)), None, 3),
        hidden=Categoric([16, 32, 64, 128], None, 32),
        hidden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        # edge_droprate=Categoric([0., 0.2, 0.4, 0.5, 0.6], None, 0.0),
        # edge_droprate=Categoric([0.0, 0.15, 0.3, 0.45, 0.6], None, 0.0),
        edge_droprate=Categoric([0.0], None, 0.0),
        # edge_droprate=Categoric([0.0, 0.1, 0.2, 0.3], None, 0.0),
        # feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        # loss_type=Categoric(["focal_loss", "ce_loss"], None, "ce_loss"),
        loss_type=Categoric(["ce_loss"], None, "ce_loss"),
        res_type=Categoric([0., 1., 2.], None, 0.),
        # res_type=Categoric([0., 1., 2., 3.], None, 0.)
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config,
                 non_hpo_config,
                 ):
        self._device = device
        self._num_class = num_class
        self.model = GCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16), config.get("hidden_droprate", 0.5),
            config.get("edge_droprate", 0.0), config.get("res_type", 0.0),
            non_hpo_config.get("directed", False)).to(device)
            # False).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.loss_type = config.get("loss_type", "focal_loss")
        self.fl_loss = FocalLoss(config.get("gamma", 2), non_hpo_config.get("label_alpha", []), device, is_minority=non_hpo_config.get("is_minority", None))

    @classmethod
    def ensure_memory_safe(cls,
                           num_nodes,
                           num_edges,
                           n_class,
                           features_num,
                           directed):
        max_hidden_units = max(int(16000000000 / 3 / 4 / (num_nodes+num_edges)), 1)
        if directed:
            max_hidden_units = int(max_hidden_units/3)
        hidden_list = GCNAlgo.hyperparam_space['hidden'].categories
        if max_hidden_units < max(hidden_list):
            new_hidden_list = []
            for h in hidden_list:
                if h <= max_hidden_units:
                    new_hidden_list.append(h)
            if len(new_hidden_list) == 0:
                new_hidden_list = [max_hidden_units]
            GCNAlgo.hyperparam_space['hidden'].categories = new_hidden_list
            if GCNAlgo.hyperparam_space['hidden'].default_value not in new_hidden_list:
                GCNAlgo.hyperparam_space['hidden'].default_value = new_hidden_list[0]
            return True
        return False

    @classmethod
    def is_memory_safe(cls,
                       num_nodes,
                       num_edges,
                       n_class,
                       features_num,
                       config,
                       non_hpo_config=None):
        """estimate the (gpu/cpu) memory consumption"""
        num_bytes = GCN.bottleneck_tensor_size(
                        num_nodes, num_edges, n_class, features_num,
                        config.get("num_layers", 2), config.get("hidden", 16),
                        config.get("hidden_droprate", 0.5),
                        config.get("edge_droprate", 0.0))
        # very reservative in this way
        # `MessageParsing` provides an implementation of `propagate()`
        # where tensors of `bottleneck_tensor_size` are allocated twice during
        # the lifecycle of this `propagate()`. We consider the worst case
        # where the first has been allocated and `torch.cuda.cached_memory()`
        # is just less than `bottleneck_tensor_size`. In this case, torch
        # tries to request the `bottleneck_tensor_size` additional memory
        # from cuda. If there is no free memory larger than what required,
        # OOM would be triggered. In a word, worst case we need three times of
        # the `bottleneck_tensor_size`. This very ad-hoc estimation serves as
        # a workaround here.
        is_safe = 3*num_bytes <= 16000000000
        if is_safe:
            logger.debug("=== a safe config {} ===".format(config))
        else:
            logger.warn("=== an unsafe config {} ===".format(config))
        return is_safe
