from torch_geometric.datasets import Amazon, CoraFull, Planetoid
from torch_geometric import utils

import os.path as osp
import numpy as np
import torch

import os
os.system("pip install ogb")
from ogb.nodeproppred import PygNodePropPredDataset


write_f_tags = [True, True, True, True, True]
write_f_tags = [not i for i in write_f_tags]
write_f_tags[0] = True
write_f_tags[3] = True
CONFIG_F, EDGE_F, FEA_F, LABEL_F, NODE_F = write_f_tags


def write_nodes_and_labels(data, nodes_idx, f_path):
    with open(f_path, "w", encoding="utf-8") as f_out:
        f_out.write("node_index\tlabel\n")
        labels = data.y[nodes_idx]

        for i in range(nodes_idx.shape[0]):
            if labels[i].ndim == 1:
                # for multi-class datasets, we simply use the first label
                f_out.write("{}\t{}\n".format(nodes_idx[i], labels[i][0]))
            elif labels[i].ndim == 0:
                f_out.write("{}\t{}\n".format(nodes_idx[i], labels[i]))
            else:
                raise ValueError("Invalid label shape")


def write_train_data_f(data, train_idx, test_idx, f_path, time_budget):
    if CONFIG_F:
        # config.yml
        config_f = f_path + "config.yml"
        with open(config_f, "w", encoding="utf-8") as f_out:
            f_out.write("n_class: {}\n".format(torch.unique(data.y).shape[0]))
            f_out.write("schema:\n")
            for i in range(data.x.shape[1]):
                fea_type = "num"
                f_out.write("  f{}: {}\n".format(i, fea_type))
            f_out.write("time_budget: {}\n".format(time_budget))


    if EDGE_F:
        # edge.tsv
        edge_f = f_path + "edge.tsv"
        with open(edge_f, "w", encoding="utf-8") as f_out:
            f_out.write("src_idx\tdst_idx\tedge_weight\n")
            src, dst = data.edge_index
            weight = np.ones(data.edge_index.shape[1]) if data.edge_attr is None else data.edge_attr
            for i in range(data.edge_index.shape[1]):
                f_out.write("{}\t{}\t{}\n".format(src[i], dst[i], weight[i]))

    if FEA_F:
       # feature.tsv
       fea_f = f_path + "feature.tsv"
       with open(fea_f, "w", encoding="utf-8") as f_out:
           head = "node_index\t"
           for i in range(data.num_features):
               head += "f{}\t".format(i)
           f_out.write(head + "\n")
           for i in range(data.num_nodes):
               line = "{}\t".format(i)
               fea_j = data.x[i].tolist()
               line += "\t".join(str(e) for e in fea_j) + "\t"
               f_out.write("{}\n".format(line))

    if NODE_F:
        # test_node_id.txt
        test_n_f = f_path + "test_node_id.txt"
        with open(test_n_f, "w", encoding="utf-8") as f_out:
            for id in test_idx:
                f_out.write("{}\n".format(id))

        # train_node_id.txt
        train_n_f = f_path + "train_node_id.txt"
        with open(train_n_f, "w", encoding="utf-8") as f_out:
            for id in train_idx:
                f_out.write("{}\n".format(id))

    if LABEL_F:
        # train_label.tsv
        train_label_f = f_path + "train_label.tsv"
        write_nodes_and_labels(data, train_idx, train_label_f)


def save_data_to_target_form(data, data_path, time_budget):
    f_path = data_path + "test_label.tsv"
    train_i = round(0.4 * data.num_nodes)
    all_idx = np.arange(data.num_nodes)
    np.random.shuffle(all_idx)
    train_nodes = all_idx[:train_i]
    test_nodes = all_idx[train_i:]

    if LABEL_F:
        write_nodes_and_labels(data, test_nodes, f_path)

    if not os.path.isdir(data_path + "train.data"):
        os.makedirs(data_path + "train.data")
    f_path = data_path + "train.data/"
    write_train_data_f(data, train_nodes, test_nodes, f_path, time_budget)


def main():
    dataset_names = ['ogbn-arxiv', 'computers', 'photo', 'pubmed', 'cora_full']
    datasets_all = [PygNodePropPredDataset, Amazon, Amazon, Planetoid, CoraFull]
    time_budgets = [200, 100, 100, 100, 150]
    dataset_name, datasets = 'ogbn-arxiv', PygNodePropPredDataset
    dataset_name, datasets = 'computers', Amazon
    dataset_name, datasets = 'photo', Amazon
    dataset_name, datasets = 'pubmed', Planetoid
    dataset_name, datasets = 'cora_full', CoraFull
    time_budget = 100

    for i in range(5):
        dataset_name = dataset_names[i]
        datasets = datasets_all[i]
        time_budget = time_budgets[i]
        generate_datasets(dataset_name, datasets, time_budget)


def generate_datasets(dataset_name, datasets, time_budge):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
    data_path = "./data/external/{dataset_name}/".format(dataset_name=dataset_name)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    if datasets is Amazon:
        dataset = datasets(path, dataset_name)
    elif datasets is CoraFull:
        dataset = datasets(path)
    elif datasets is Planetoid:
        dataset = datasets(path, dataset_name)
    elif datasets is PygNodePropPredDataset:
        dataset = datasets(dataset_name)
    else:
        raise ValueError("un-supported dataset")
    data = dataset[0]
    print(dataset_name)
    if utils.is_undirected(data.edge_index):
        print("This is un-diredcted graph")
    else:
        print("This is diredcted graph")
    save_data_to_target_form(data, data_path, time_budge)


if __name__ == '__main__':
    main()
