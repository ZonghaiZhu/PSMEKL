# coding:utf-8
import numpy as np
import networkx as nx
import numpy.linalg as LA
from node2vec import Node2Vec
import os
from torch_sparse import SparseTensor
from gensim.models import Word2Vec
from tqdm import tqdm
import torch, random


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

    num_nodes = data.num_nodes
    G = nx.Graph()
    edges = data.edge_index
    for i in range(data.num_edges):
        G.add_edge(edges[0, i].item(), edges[1, i].item())

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    return_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))

    if emb_type == 'PSMEKL' or emb_type == 'SMEKL':
        coms = nx.community.louvain_communities(G=G)
        num_classes = len(coms)
        train_idx, labels = [], []
        for i in range(num_classes):
            real_idx = list(coms[i])
            len_nodes = len(real_idx)
            train_idx.extend(real_idx)
            labels.extend([i]*len_nodes)

    ###########################
    # adj
    ###########################
    if emb_type == 'adj':
        Temp_adj = adj
        for l in range(emb_layer - 1):
            Temp_adj = Temp_adj @ adj
        node_feats = Temp_adj.to_dense()

        return node_feats, return_adj

    ###########################
    # shared
    ###########################
    if emb_type == 'shared':
        node_feats = np.ones((num_nodes, emb_dim))

        return node_feats, return_adj

    ###########################
    # degree
    ###########################
    if emb_type == 'degree':
        node_feats = adj.sum(-1)

        return node_feats, return_adj

    ###########################
    # pagerank
    ###########################
    if emb_type == 'pagerank':
        saved_path = os.path.join('embedding_features', emb_type)
        try:
            v = np.load(os.path.join(saved_path, data.name) + '.npy')
            print(v.shape)
        except:
            pageranks = nx.pagerank(G)
            v = np.empty((num_nodes, 1))
            for i, p in pageranks.items():
                v[i] = p
            np.save(os.path.join(dir, data.name + '.npy'), v)

        return v, return_adj

    ###########################
    # eigen
    ###########################
    if emb_type == 'eigen' or emb_type == 'eigen_norm':
        saved_path = os.path.join('embedding_features', emb_type)
        try:
            if emb_type == "eigen":
                v = np.load(os.path.join(saved_path, data.name + '_' + str(emb_layer) + '_' + str(emb_dim)) + '.npy')
            else:
                v = np.load(
                    os.path.join(saved_path, data.name + '_' + 'degree_normalized' + '_' + str(emb_layer) + '_' + str(emb_dim) + '.npy'))
            print(v.shape)
        except:
            adj_matrix = nx.to_numpy_array(G)
            Temp_adj = adj_matrix
            for l in range(emb_layer - 1):
                Temp_adj = Temp_adj @ adj_matrix
            # normalize adjacency matrix with degree
            if emb_type == "eigen_norm":
                sum_of_rows = Temp_adj.sum(axis=1)
                Temp_adj = Temp_adj / sum_of_rows[:, None]
            print("start computing eigen vectors")
            w, v = LA.eig(Temp_adj)
            indices = np.argsort(w)[::-1]
            v = v.transpose()[indices]

            if emb_type == "eigen":
                np.save(os.path.join(dir, data.name + '_' + str(emb_layer) + '_' + str(emb_dim) + '.npy'),
                        v[:emb_dim])
            else:
                np.save(os.path.join(saved_path, data.name + '_' + 'degree_normalized' + '_' + str(emb_layer) + '_' + str(emb_dim) + '.npy'),
                        v[:emb_dim])

        node_feats = np.zeros((num_nodes, emb_dim))

        for i in range(num_nodes):
            for j in range(emb_dim):
                node_feats[i, j] = v[j, i]

        return node_feats, return_adj

    ###########################
    # node2vec
    ###########################
    if emb_type == 'node2vec':
        saved_path = os.path.join('embedding_features', emb_type)
        try:
            v = np.load(os.path.join(saved_path, data.name) + '_' + str(emb_dim) + '.npy')
            print(v.shape)
        except:
            node2vec = Node2Vec(G, dimensions=emb_dim, num_walks=40, workers=4)
            molel = node2vec.fit(window=10, hs=0)
            v = molel.wv.vectors
            np.save(os.path.join(dir, data.name + '_' + str(emb_dim) + '.npy'), v)

        return v, return_adj

    ###########################
    # deepwalk
    ###########################
    if emb_type == 'deepwalk':

        def get_randomwalk(node, path_length):
            random_walk = [node]
            for i in range(path_length - 1):
                temp = list(G.neighbors(node))
                temp = list(set(temp) - set(random_walk))
                if len(temp) == 0:
                    break
                random_node = random.choice(temp)
                random_walk.append(random_node)
                node = random_node

            return random_walk

        saved_path = os.path.join('embedding_features', emb_type)
        try:
            v = np.load(os.path.join(saved_path, data.name) + '_' + str(emb_dim) + '.npy')
            print(v.shape)
        except:
            all_nodes = list(G.nodes())
            random_walks = []
            for n in tqdm(all_nodes):
                for i in range(80):
                    random_walks.append(get_randomwalk(n, 40))

            # train word2vec model
            model = Word2Vec(vector_size=emb_dim, window=10)

            model.build_vocab(random_walks, progress_per=2)
            model.train(random_walks, total_examples=model.corpus_count, epochs=100, report_delay=1)
            v = model.wv.vectors

            np.save(os.path.join(dir, data.name + '_' + str(emb_dim) + '.npy'), v)

        return v, return_adj

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
    def __init__(self, node_lists, emb_dim, labels, train_idx, lr=0.01, epochs=100):
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