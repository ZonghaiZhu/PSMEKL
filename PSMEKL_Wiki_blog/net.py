# coding:utf-8
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, 'sum'))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, 'sum'))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x.log_softmax(dim=-1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn=torch.nn.Linear(in_channels, hidden_channels)))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn=torch.nn.Linear(hidden_channels, hidden_channels)))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GINConv(nn=torch.nn.Sequential(torch.nn.Linear(hidden_channels, out_channels))))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x.log_softmax(dim=-1)


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_layers = args.gnn_layer
        self.emb_dim = args.gnn_emb_dim
        self.hidden_channels = args.gnn_emb_dim
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.dropout = args.dropout
        if args.emb_type == 'degree':
            self.embedding = torch.nn.Embedding(100, self.emb_dim)
            self.num_features = self.emb_dim

        if args.gnn == 'sage':
            self.gnn = SAGE(self.num_features, self.hidden_channels,
                            self.num_classes, self.num_layers,
                            self.dropout).to(args.device)
        elif args.gnn == 'gcn':
            self.gnn = GCN(self.num_features, self.hidden_channels,
                           self.num_classes, self.num_layers,
                           self.dropout).to(args.device)
        elif args.gnn == 'gin':
            self.gnn = GIN(self.num_features, self.hidden_channels,
                           self.num_classes, self.num_layers,
                           self.dropout).to(args.device)

    def forward(self, x, adj, emd_type=None):
        if emd_type == 'degree':
            x = x.to(torch.long) % 100
            x = self.embedding(x)

        if emd_type == 'pagerank':
            x = x * 100

        return self.gnn(x, adj)
