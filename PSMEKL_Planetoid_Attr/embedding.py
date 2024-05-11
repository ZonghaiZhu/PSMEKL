# coding:utf-8
import numpy as np
import networkx as nx
import numpy.linalg as LA
from node2vec import Node2Vec
import os, random, torch
from torch_sparse import SparseTensor
from gensim.models import Word2Vec
from tqdm import tqdm


# The following function contains all comparison algorithms
def node_emb(data, emb_dim, emb_layer, emb_type, num_subsets):
    # set random seed for node embedding
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.random.manual_seed(0)

    dir = 'embedding_features/' + emb_type
    if not os.path.exists(dir):
        if not os.path.exists('embedding_features'):
            os.makedirs('embedding_features')
        os.makedirs(dir)

    num_nodes = data[0].num_nodes
    G = nx.Graph()
    edges = data[0].edge_index
    for i in range(data[0].num_edges):
        G.add_edge(edges[0, i].item(), edges[1, i].item())

    if len(G.nodes) != num_nodes:
        conained_nodes = set(list(edges[0].numpy())+list(edges[1].numpy()))
        rest_nodes = set(range(num_nodes)) - conained_nodes
        for i in rest_nodes:
            G.add_edge(i, i)

    row, col = data[0].edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    return_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # adj = adj.set_diag()

    if emb_type == 'PSMEKL' or emb_type == 'SMEKL':
        coms = nx.community.louvain_communities(G=G)
        num_classes = len(coms)
        train_idx, labels = [], []
        for i in range(num_classes):
            real_idx = list(coms[i])
            len_nodes = len(real_idx)
            train_idx.extend(real_idx)
            labels.extend([i] * len_nodes)

    ###########################
    # original
    ###########################
    if emb_type == 'Original':
        node_feats = data.x

        return node_feats, return_adj

    ###########################
    # EKM
    ###########################
    if emb_type == 'EKM':

        Temp_adj = adj
        for l in range(emb_layer - 1):
            Temp_adj = Temp_adj @ adj
        node_feats = Temp_adj.to_dense()
        par = aveRBFPar(node_feats, num_nodes)
        implicitKernel = kernel_mapping(node_feats, node_feats, 'rbf', par)
        w, v = LA.eig(implicitKernel)
        w = np.float32(w.real)
        v = np.float32(v.real)
        index = np.where(w >= 1e-3)[0]

        P = v[:, index]
        R = np.diag(np.power(w[index], -0.5))

        emp_train = implicitKernel @ P @ R

        return emp_train, return_adj

    ###########################
    # Multi-subsets EKM
    ###########################
    if emb_type == 'MEKL':
        node_per_subset = int(num_nodes / num_subsets)
        Temp_adj = adj
        for l in range(emb_layer - 1):
            Temp_adj = Temp_adj @ adj
        node_feats = Temp_adj.to_dense()

        node_lists = []
        ind_list = list(range(num_nodes))
        random.shuffle(ind_list)
        for i in range(num_subsets):
            if i != num_subsets - 1:
                inds = ind_list[i * node_per_subset: (i + 1) * node_per_subset]
            else:
                inds = ind_list[i * node_per_subset:]
            temp_feats = node_feats[inds]
            par = aveRBFPar(temp_feats, len(inds))
            implictKernel = kernel_mapping(temp_feats, temp_feats, 'rbf', par)
            w, v = LA.eig(implictKernel)
            w = np.float32(w.real)
            v = np.float32(v.real)
            index = np.where(w >= 1e-3)[0]

            P = v[:, index]
            R = np.diag(np.power(w[index], -0.5))

            implictKernel1 = kernel_mapping(node_feats, temp_feats, 'rbf', par)
            emp_train = implictKernel1 @ P @ R

            node_lists.append(emp_train)
        emp_train = np.concatenate(node_lists, axis=-1)

        return emp_train, return_adj

    ###########################
    # Consistent Multi-subsets EKM
    ###########################
    if emb_type == 'PMEKL':
        node_per_subset = int(num_nodes / num_subsets)
        # Temp_adj = adj
        # for l in range(emb_layer - 1):
        #     Temp_adj = Temp_adj @ adj
        # node_feats = Temp_adj.to_dense()
        node_feats = data[0].x

        node_lists = []
        ind_list = list(range(num_nodes))
        random.shuffle(ind_list)
        for i in range(num_subsets):
            if i != num_subsets - 1:
                inds = ind_list[i * node_per_subset: (i + 1) * node_per_subset]
            else:
                inds = ind_list[i * node_per_subset:]
            temp_feats = node_feats[inds]
            par = aveRBFPar(temp_feats, len(inds))
            implictKernel = kernel_mapping(temp_feats, temp_feats, 'rbf', par)
            w, v = LA.eig(implictKernel)
            w = np.float32(w.real)
            v = np.float32(v.real)
            index = np.where(w >= 1e-3)[0]

            P = v[:, index]
            R = np.diag(np.power(w[index], -0.5))

            implictKernel1 = kernel_mapping(node_feats, temp_feats, 'rbf', par)
            emp_train = implictKernel1 @ P @ R

            node_lists.append(emp_train.to('cuda'))

        model = PModel(node_lists, emb_dim, lr=0.1, epochs=100).to('cuda')
        emp_train = model.fit()
        emp_train = torch.tensor(emp_train.clone().detach(), dtype=torch.float32)
        del model
        return emp_train, return_adj

    ###########################
    # Labeled Multi-subsets EKM
    ###########################
    if emb_type == 'SMEKL':
        node_per_subset = int(num_nodes / num_subsets)
        # Temp_adj = adj
        # for l in range(emb_layer - 1):
        #     Temp_adj = Temp_adj @ adj
        # node_feats = Temp_adj.to_dense()
        node_feats = data[0].x

        node_lists = []
        ind_list = list(range(num_nodes))
        random.shuffle(ind_list)
        for i in range(num_subsets):
            if i != num_subsets - 1:
                inds = ind_list[i * node_per_subset: (i + 1) * node_per_subset]
            else:
                inds = ind_list[i * node_per_subset:]
            temp_feats = node_feats[inds]
            par = aveRBFPar(temp_feats, len(inds))
            implictKernel = kernel_mapping(temp_feats, temp_feats, 'rbf', par)
            w, v = LA.eig(implictKernel)
            w = np.float32(w.real)
            v = np.float32(v.real)
            index = np.where(w >= 1e-3)[0]

            P = v[:, index]
            R = np.diag(np.power(w[index], -0.5))

            implictKernel1 = kernel_mapping(node_feats, temp_feats, 'rbf', par)
            emp_train = implictKernel1 @ P @ R

            node_lists.append(emp_train.to('cuda'))

        model = SModel(node_lists, emb_dim, labels, train_idx, lr=0.1, epochs=100).to('cuda')
        emp_train = model.fit()
        emp_train = torch.tensor(emp_train.clone().detach(), dtype=torch.float32)
        del model
        return emp_train, return_adj


    ###########################
    # Consistent and Labeled Multi-subsets EKM
    ###########################
    if emb_type == 'PSMEKL':
        node_per_subset = int(num_nodes / num_subsets)
        # Temp_adj = adj
        # for l in range(emb_layer - 1):
        #     Temp_adj = Temp_adj @ adj
        # node_feats = Temp_adj.to_dense()
        node_feats = data[0].x

        node_lists = []
        ind_list = list(range(num_nodes))
        random.shuffle(ind_list)
        for i in range(num_subsets):
            if i != num_subsets - 1:
                inds = ind_list[i * node_per_subset: (i + 1) * node_per_subset]
            else:
                inds = ind_list[i * node_per_subset:]
            temp_feats = node_feats[inds]
            par = aveRBFPar(temp_feats, len(inds))
            implictKernel = kernel_mapping(temp_feats, temp_feats, 'rbf', par)
            w, v = LA.eig(implictKernel)
            w = np.float32(w.real)
            v = np.float32(v.real)
            index = np.where(w >= 1e-3)[0]

            P = v[:, index]
            R = np.diag(np.power(w[index], -0.5))

            implictKernel1 = kernel_mapping(node_feats, temp_feats, 'rbf', par)
            emp_train = implictKernel1 @ P @ R

            node_lists.append(emp_train.to('cuda'))

        model = PSModel(node_lists, emb_dim, labels, train_idx, lr=0.1, epochs=100).to('cuda')
        emp_train = model.fit()
        emp_train = torch.tensor(emp_train.clone().detach(), dtype=torch.float32)
        del model
        return emp_train, return_adj


class PModel(torch.nn.Module):
    def __init__(self, node_lists, emb_dim, lr=0.01, epochs=100):
        super(PModel, self).__init__()
        self.node_lists = node_lists
        self.num_subsets = len(node_lists)
        self.num_nodes = node_lists[0].shape[0]
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.net = ConsistNet(self.node_lists, self.emb_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self):
        for epoch in range(self.epochs):
            self.net.zero_grad()
            outputs = self.net(self.node_lists)
            mean_outputs = torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze(1)
            l = 0
            for i in range(self.num_subsets):
                temp = outputs[i].squeeze(1) - mean_outputs
                l += torch.trace(temp @ temp.T) / self.num_nodes
            l.backward()
            self.optimizer.step()

            print(('Epoch:[{}/{}], Train_Loss:{:.4f}').format(epoch, self.epochs, l))

        return torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze(1)


class SModel(torch.nn.Module):
    def __init__(self, node_lists, emb_dim, labels, train_idx, lr=0.01, epochs=100):
        super(SModel, self).__init__()
        self.node_lists = node_lists
        self.num_subsets = len(node_lists)
        self.num_nodes = node_lists[0].shape[0]
        self.emb_dim = emb_dim
        self.train_idx = train_idx
        self.labels = labels
        self.epochs = epochs
        self.mlp1 = torch.nn.ModuleList()
        self.mlp2 = torch.nn.ModuleList()
        for i in range(self.num_subsets):
            self.mlp1.append(torch.nn.Linear(node_lists[i].shape[-1], self.emb_dim).to('cuda'))
            self.mlp2.append(torch.nn.Linear(self.emb_dim, len(set(labels))).to('cuda'))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.mlp1.parameters(), lr=lr)

    def fit(self):
        for epoch in range(self.epochs):
            self.zero_grad()
            outputs = []
            loss = 0
            for i in range(self.num_subsets):
                temp = self.mlp1[i](self.node_lists[i]).unsqueeze(dim=1)
                outputs.append(temp)
                temp_cls = self.mlp2[i](temp.squeeze(1))
                loss += self.criterion(temp_cls[self.train_idx], torch.LongTensor(self.labels).to('cuda'))
            loss.backward()
            self.optimizer.step()

            print(('Epoch:[{}/{}], Train_Loss:{:.4f}').format(epoch, self.epochs, loss))

        return torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze(1)


class PSModel(torch.nn.Module):
    def __init__(self, node_lists, emb_dim, labels, train_idx, lr=0.1, epochs=100):
        super(PSModel, self).__init__()
        self.node_lists = node_lists
        self.num_subsets = len(node_lists)
        self.num_nodes = node_lists[0].shape[0]
        self.emb_dim = emb_dim
        self.train_idx = train_idx
        self.labels = labels
        self.epochs = epochs
        self.net = ConsistNet(self.node_lists, self.emb_dim)
        self.mlp = torch.nn.ModuleList()
        for i in range(self.num_subsets):
            self.mlp.append(torch.nn.Linear(self.emb_dim, len(set(labels))).to('cuda'))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self):
        for epoch in range(self.epochs):
            self.net.zero_grad()
            outputs = self.net(self.node_lists)
            mean_outputs = torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze(1)
            l = 0
            for i in range(self.num_subsets):
                temp = outputs[i].squeeze(1) - mean_outputs
                temp_cls = self.mlp[i](outputs[i].squeeze(1))
                loss = self.criterion(temp_cls[self.train_idx], torch.LongTensor(self.labels).to('cuda'))
                l += 0.5 * (torch.trace(temp @ temp.T) / self.num_nodes + loss)
            l.backward()
            self.optimizer.step()

            print(('Epoch:[{}/{}], Train_Loss:{:.4f}').format(epoch, self.epochs, l))

        return torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze(1)


class ConsistNet(torch.nn.Module):
    def __init__(self, node_lists, emb_dim):
        super(ConsistNet, self).__init__()
        self.mlps = torch.nn.ModuleList()
        self.num_subsets = len(node_lists)
        self.emb_dim = emb_dim
        for i in range(self.num_subsets):
            self.mlps.append(torch.nn.Linear(node_lists[i].shape[-1], emb_dim, bias=False))
            self.mlps[i].weight.data = torch.rand(emb_dim, node_lists[i].shape[-1]) + 0.5  # must avoid outputs to 0

    def forward(self, node_lists):
        outputs = []

        for i in range(self.num_subsets):
            outputs.append(self.mlps[i](node_lists[i]).unsqueeze(dim=1))

        return outputs  # , w


def kernel_mapping(mat_train, mat_test, kernelType, par):
    if kernelType == 'rbf':
        TrainSampleNum = mat_train.shape[0]
        TestSampleNum = mat_test.shape[0]
        mat_temp = (mat_train.pow(2).sum(dim=-1) * torch.ones((TestSampleNum, TrainSampleNum))).T + \
                   mat_test.pow(2).sum(dim=-1) * torch.ones((TrainSampleNum, TestSampleNum)) - \
                   2 * mat_train @ mat_test.T
        mat_kernel = np.exp(-mat_temp / (2 * par.pow(2)))

    return mat_kernel


def aveRBFPar(data, size):
    temp = data.pow(2).sum(dim=-1) * torch.ones((size, size))
    mat = temp.T + temp - 2 * data @ data.T
    mat_mean = mat.sum() / (size * size)
    par = mat_mean.sqrt()

    return par