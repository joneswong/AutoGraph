from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .gcn_algo import GNNAlgo, FocalLoss
from .gnns import *

from spaces import Categoric, Numeric

import dgl
import logging

logger = logging.getLogger('code_submission')


class AdaGCN(torch.nn.Module):
    def __init__(self,
                 num_class,
                 features_num,
                 num_layers=2,
                 hidden=16,
                 hidden_droprate=0.5, edge_droprate=0.0, res_type=0.0, directed=False, shared_tau=True):

        super(AdaGCN, self).__init__()
        self.first_lin = Linear(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        if directed:
            # self.convs.append(DglGCNConv(hidden, hidden * 2))
            # hidden = hidden * 2
            self.convs.append(DglGCNConv(features_num, hidden * 2))
            hidden = hidden * 2
        else:
            # self.convs.append(DglGCNConv(hidden, hidden))
            self.convs.append(DglGCNConv(features_num, hidden))
        for i in range(num_layers - 1):
            self.convs.append(GatedLayer(hidden, hidden))
        self.lin2 = Linear(hidden, num_class)
        self.hidden_droprate = hidden_droprate
        self.edge_droprate = edge_droprate
        self.res_type = res_type
        self.directed = directed
        self.g = dgl.DGLGraph()

        self.weight_y = nn.Linear(hidden, num_class)
        self.global_tau_1 = nn.Parameter(torch.zeros((1,)))
        self.global_tau_2 = nn.Parameter(torch.zeros((1,)))
        self.shared_tau = shared_tau
        self.num_class = num_class

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def init_weight_y(self, feats, labels):
        # initial weight_y is obtained by linear regression
        A = torch.mm(feats.t(), feats) + 1e-05 * torch.eye(feats.size(1), device=feats.device)
        # (feats, feats)

        categorical_y = labels.view(-1, 1)
        N, C = feats.size(0), self.num_class
        labels_one_hot = torch.zeros([N, C], dtype=torch.float32, device=categorical_y.device)
        labels_one_hot.scatter_(1, categorical_y, 1)


        self.first_weight_y = nn.Parameter(torch.mm(torch.mm(torch.cholesky_inverse(A), feats.t()), labels_one_hot),
                                     requires_grad=False)
        nn.init.constant_(self.global_tau_1, 1 / 2)
        nn.init.constant_(self.global_tau_2, 1 / 2)
        return

    def forward(self, data):
        z = torch.FloatTensor([1.0, ]).cuda()
        list_z = []
        is_real_weighted_graph = data["real_weight_edge"]
        # norm_type = "right" if data["directed"] else "both"
        # another directed GCN used in R-GCN, have not achieve improvements on feedback dataset 3
        # for conv in self.convs:
        #     conv._norm = norm_type

        if self.g.number_of_nodes() == 0:  # first forward, build the graph once
            for lidx, conv in enumerate(self.convs):
                if lidx != 0:
                    conv.init_layer_norm(data.num_nodes)

        if self.edge_droprate != 0.0:
            x = data.x
            edge_index, edge_weight = dropout_adj(data.edge_index, data.edge_weight, self.edge_droprate)
            self.g.clear()
            self.g.add_nodes(data.num_nodes)
            self.g.add_edges(edge_index[0], edge_index[1])
            self.g.add_edges(self.g.nodes(), self.g.nodes())  # add self-loop to avoid invalid normalizer
            if is_real_weighted_graph:
                self.g.edata["weight"] = torch.cat(
                    (edge_weight, torch.ones(self.g.number_of_nodes(), device=data.x.device)))
                # weighted degrees
                self.g.update_all(fn.copy_e("weight", "e_w"), fn.sum("e_w", "in_degs"))
                self.g.ndata['norm'] = torch.pow(self.g.ndata["in_degs"], -0.5).unsqueeze(1)
            else:
                degs = self.g.in_degrees().to(x.device).float().clamp(min=1)
                self.g.ndata["in_degs"] = degs
                self.g.ndata['norm'] = torch.pow(degs, -0.5)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
            if self.g.number_of_nodes() == 0:  # first forward, build the graph once
                self.g.add_nodes(data.num_nodes)
                self.g.add_edges(edge_index[0], edge_index[1])
                self.g.add_edges(self.g.nodes(), self.g.nodes())  # add self-loop to avoid invalid normalizer
                if is_real_weighted_graph:
                    self.g.edata["weight"] = torch.cat(
                        (edge_weight, torch.ones(self.g.number_of_nodes(), device=data.x.device)))
                    # weighted degrees
                    self.g.update_all(fn.copy_e("weight", "e_w"), fn.sum("e_w", "in_degs"))
                    self.g.ndata['norm'] = torch.pow(self.g.ndata["in_degs"], -0.5).unsqueeze(1)
                else:
                    degs = self.g.in_degrees().to(x.device).float().clamp(min=1)
                    self.g.ndata["in_degs"] = degs
                    self.g.ndata['norm'] = torch.pow(degs, -0.5)

                # first forward, init the weight_y
                all_train_mask = data.train_mask
                self.init_weight_y(data.x[all_train_mask], data.y[all_train_mask])

        for lidx, layer in enumerate(self.convs):
            if lidx == 0:
                # logits in first layer are calculated by first_weight_y
                logits = F.softmax(torch.mm(x, self.first_weight_y), dim=1)
                x = layer(self.g, x, real_weighted_g=is_real_weighted_graph)
            else:
                x = F.dropout(x, p=self.hidden_droprate, training=self.training)
                logits = F.softmax(self.weight_y(x), dim=1)
                x, z = layer(self.g, x, logits, old_z=z,
                             shared_tau=self.shared_tau, tau_1=self.global_tau_1, tau_2=self.global_tau_2)
                list_z.append(z)

        # if self.res_type == 0.0:
        #     x = F.relu(self.first_lin(x))
        #     x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        #     for conv in self.convs:
        #         x = F.relu(conv(self.g, x, real_weighted_g=is_real_weighted_graph))
        #     x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        #     x = self.lin2(x)
        # else:
        #     x = F.relu(self.first_lin(x))
        #     x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        #     x_list = [] if self.directed else [x]
        #     for conv in self.convs:
        #         x = F.relu(conv(self.g, x, real_weighted_g=is_real_weighted_graph))
        #         x_list.append(x)
        #     if self.res_type == 1.0:
        #         x = x + x_list[0]
        #     elif self.res_type == 2.0:
        #         x = torch.sum(torch.stack(x_list, 0), 0)
        #     x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        #     x = self.lin2(x)

        # return F.log_softmax(x, dim=-1)
        # due to focal loss: return the logits, put the log_softmax operation into the GNNAlgo
        all_z = torch.stack(list_z, dim=1)  # (n_nodes, n_layers)

        return x, all_z

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
        return 4 * (num_nodes + num_edges) * hidden


class SGCNAlgo(GNNAlgo):
    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 16),
        hidden_droprate=Numeric((), np.float32, 0.2, 0.6, 0.5),
        lr=Numeric((), np.float32, 1e-5, 1e-2, 5e-3),
        weight_decay=Numeric((), np.float32, .0, 1e-3, 5e-4)
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config):
        self._device = device
        self._num_class = num_class
        self.model = SGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num


class SplineGCNAlgo(GNNAlgo):
    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 100], None, 16),
        hiden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Categoric([0.0, 0.2, 0.5], None, 0.0),
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        loss_type=Categoric(["focal_loss", "ce_loss"], None, "focal_loss"),
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
        self._num_layer = config.get("num_layers", 2)
        self.model = SplineGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("dim", 1),
            config.get("kernel_size", 2), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.loss_type = config.get("loss_type", "focal_loss")
        self.fl_loss = FocalLoss(config.get("gamma", 2), non_hpo_config.get("label_alpha", []), device)

    @classmethod
    def is_memory_safe(cls,
                       num_nodes,
                       num_edges,
                       n_class,
                       features_num,
                       config,
                       non_hpo_config=None):
        # (daoyuan) TO DO: implement
        return True

    @classmethod
    def ensure_memory_safe(cls,
                           num_nodes,
                           num_edges,
                           n_class,
                           features_num,
                           directed):
        # (daoyuan) TO DO: implement
        pass


class SplineGCN_APPNPAlgo(GNNAlgo):
    hyperparam_space = dict(
        num_layers=Categoric(list(range(2, 5)), None, 2),
        hidden=Categoric([16, 32, 64, 100], None, 16),
        hiden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        dim=Categoric([1], None, 1),
        kernel_size=Categoric([2, 3, 4], None, 2),
        edge_droprate=Categoric([0.0, 0.2, 0.5], None, 0.0),
        feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # todo (daoyuan): add pair_norm and batch_norm
        loss_type=Categoric(["focal_loss", "ce_loss"], None, "focal_loss"),
        iter_k=Categoric([10, 20, 40], None, 10),
        tele_prob=Categoric([0.2, 0.5, 0.7], None, 0.5),
    )

    def __init__(self,
                 num_class,
                 features_num,
                 device,
                 config,
                 focal_loss=False):
        self._device = device
        self._num_class = num_class
        self.model = SplineGCN(
            config.get("num_layers", 2), config.get("hidden", 16), features_num, num_class,
            config.get("hidden_droprate", 0.5), config.get("dim", 1),
            config.get("kernel_size", 2), config.get("edge_droprate", 0.0)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.focal_loss = focal_loss



class AdaGCNAlgo(GNNAlgo):

    hyperparam_space = dict(
        num_layers=Categoric(list(range(1, 4)), None, 2),
        hidden=Categoric([16, 32, 64, 128], None, 32),
        hidden_droprate=Categoric([0.3, 0.4, 0.5, 0.6], None, 0.5),
        lr=Categoric([5e-4, 1e-3, 2e-3, 5e-3, 1e-2], None, 5e-3),
        weight_decay=Categoric([0., 1e-5, 5e-4, 1e-2], None, 5e-4),
        # edge_droprate=Categoric([0., 0.2, 0.4, 0.5, 0.6], None, 0.0),
        # edge_droprate=Categoric([0.0, 0.15, 0.3, 0.45, 0.6], None, 0.0),
        edge_droprate=Categoric([0.0], None, 0.0),
        # edge_droprate=Categoric([0.0, 0.1, 0.2, 0.3], None, 0.0),
        # feature_norm=Categoric(["no_norm", "graph_size_norm"], None, "no_norm"),
        # loss_type=Categoric(["focal_loss", "ce_loss"], None, "ce_loss"),
        loss_type=Categoric(["ce_loss"], None, "ce_loss"),
        res_type=Categoric([0., 1., 2.0], None, 0.)
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
        self.model = AdaGCN(
            num_class, features_num, config.get("num_layers", 2),
            config.get("hidden", 16), config.get("hidden_droprate", 0.5),
            config.get("edge_droprate", 0.0), config.get("res_type", 0.0),
            non_hpo_config.get("directed", False)).to(device)
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 0.005),
            weight_decay=config.get("weight_decay", 5e-4))
        self._features_num = features_num
        self.loss_type = config.get("loss_type", "ce_loss")
        self.fl_loss = FocalLoss(config.get("gamma", 2), non_hpo_config.get("label_alpha", []), device,
                                 is_minority=non_hpo_config.get("is_minority", None))

    def train(self, data, data_mask, T=1.0):
        self.model.train()
        self._optimizer.zero_grad()
        if self.loss_type == "focal_loss":
            logits, z = self.model(data)
            logits = logits[data_mask]
            loss = self.fl_loss(logits, data.y[data_mask], T)
        elif self.loss_type == "ce_loss":
            logits, z = self.model(data)
            logits = logits[data_mask]
            loss = F.cross_entropy(logits, data.y[data_mask])
        else:
            raise ValueError("You give the wrong loss type. Got {}.".format(self.loss_type))
        reg = torch.norm(z * (torch.ones_like(z) - z), p=1)
        loss += reg
        loss.backward()
        self._optimizer.step()
        # We may need values other than loss for making better decisions on
        # when to stop the training course
        return {"loss": loss.item()}

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
        hidden_list = AdaGCNAlgo.hyperparam_space['hidden'].categories
        if max_hidden_units < max(hidden_list):
            new_hidden_list = []
            for h in hidden_list:
                if h <= max_hidden_units:
                    new_hidden_list.append(h)
            if len(new_hidden_list) == 0:
                new_hidden_list = [max_hidden_units]
            AdaGCNAlgo.hyperparam_space['hidden'].categories = new_hidden_list
            if AdaGCNAlgo.hyperparam_space['hidden'].default_value not in new_hidden_list:
                AdaGCNAlgo.hyperparam_space['hidden'].default_value = new_hidden_list[0]
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
        num_bytes = AdaGCN.bottleneck_tensor_size(
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
