# coding:utf-8
import os, time, argparse
import numpy as np
from torch_geometric.data import Data
import torch, random
import utils, scipy.io
from model import Model
import embedding


num_subsets_dict = {'brazil-airports': 3, 'europe-airports': 3, 'usa-airports': 3}
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', default='data/', type=str, help="data path (dictionary)")
parser.add_argument('--dataset', type=str, default="brazil-airports",
                    help='brazil-airports/europe-airports/usa-airports')

## For neural network
parser.add_argument('--gnn', type=str, default='sage', help='gcn, gin, or sage (default: gcn)')
parser.add_argument('--gnn_layer', type=int, default=15,
                    help='number of GNN message passing layers (default: 15)')
parser.add_argument('--gnn_emb_dim', type=int, default=256,
                    help='number of GNN message passing layers (default: 256)')
parser.add_argument('--emb_layer', type=int, default=1,
                    help='number of layers for node embedding of eigen-based methods (default: 1)')
parser.add_argument('--emb_type', type=str, default='node2vec',
                    help='P: adj, madj, eadj, cadj, eigen, eigen_norm, deepwalk, or node2vec | S: shared, degree, pagerank (default: IKM)')
parser.add_argument('--emb_dim', type=int, default=128, help='hidden size for node feature')

# For training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay_epoch', default=10, type=int)
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs')
parser.add_argument('--least_epoch', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--early_stop', type=int, default=30, help='patience for early stopping')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument("--run_times", type=int, default=10, help="seed for initializing training.")
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main(dataset, type):
    args = parser.parse_args()
    args.emb_type = type
    args.dataset = dataset
    num_subsets = num_subsets_dict[args.dataset]

    # prepare related data
    if args.dataset == 'usa-airports':
        with open('data/flight/labels-usa-airports.txt', 'r') as f:
            temp = f.readlines()

        idx_map = {}
        count = 0
        labels = []
        for i in range(1, len(temp)):
            idx = int(temp[i].strip('\n').split(' ')[0])
            label = int(temp[i].strip('\n').split(' ')[1])
            labels.append(label)

            idx_map[idx] = count
            count += 1

        temp = np.loadtxt('data/flight/usa-airports.edgelist', dtype=np.long).T
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                temp[i, j] = idx_map[temp[i, j]]

        # index = temp[0, :].argsort()
        # edge_list = torch.tensor(temp[:, index], dtype=torch.long)
        edge_list = torch.tensor(temp, dtype=torch.long)
        args.num_classes = len(set(labels))
        dataset = Data(name=args.dataset, y=torch.tensor(labels, dtype=torch.long), edge_index=edge_list,
                       num_nodes=len(labels), train_mask=None, val_mask=None, test_mask=None)

    if args.dataset == 'brazil-airports':
        with open('data/flight/labels-brazil-airports.txt', 'r') as f:
            temp = f.readlines()

        labels = []
        for i in range(1, len(temp)):
            label = int(temp[i].strip('\n').split(' ')[1])
            labels.append(label)

        temp = np.loadtxt('data/flight/brazil-airports.edgelist', dtype=np.long).T
        # index = temp[0, :].argsort()
        # edge_list = torch.tensor(temp[:, index], dtype=torch.long)
        edge_list = torch.tensor(temp, dtype=torch.long)
        args.num_classes = len(set(labels))
        dataset = Data(name=args.dataset, y=torch.tensor(labels, dtype=torch.long), edge_index=edge_list,
                       num_nodes=len(labels), train_mask=None, val_mask=None, test_mask=None)

    if args.dataset == 'europe-airports':
        with open('data/flight/labels-europe-airports.txt', 'r') as f:
            temp = f.readlines()

        labels = []
        for i in range(1, len(temp)):
            label = int(temp[i].strip('\n').split(' ')[1])
            labels.append(label)

        temp = np.loadtxt('data/flight/europe-airports.edgelist', dtype=np.long).T
        # index = temp[0, :].argsort()
        # edge_list = torch.tensor(temp[:, index], dtype=torch.long)
        edge_list = torch.tensor(temp, dtype=torch.long)
        args.num_classes = len(set(labels))
        dataset = Data(name=args.dataset, y=torch.tensor(labels, dtype=torch.long), edge_index=edge_list,
                       num_nodes=len(labels), train_mask=None, val_mask=None, test_mask=None)

    # prepare related documents
    print('\nSetting environment...')
    args.train_mask, args.val_mask, args.test_mask = [], [], []
    for l in range(args.num_classes):
        temp = np.where(dataset.y == l)[0]
        args.train_mask.extend(list(temp[:int(0.2 * len(temp))]))
        args.val_mask.extend(list(temp[int(0.2 * len(temp)):int(0.6 * len(temp))]))
        args.test_mask.extend(list(temp[int(0.6 * len(temp)):]))

    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = 'log/' + args.dataset + '_' + args.gnn + '_' + args.emb_type + '_' + str(args.dropout) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir)
    utils.configure_output_dir(log_dir)

    # hyperparameters: lr, dropout, weight_decay, emb_dim, permutation
    lrs = [0.001, 0.0001]
    emb_dims = [128, 256]
    if args.dataset == 'brazil-airports' and args.emb_type == 'eigen':
        emb_dims = [128]
    if args.emb_type in ['adj', 'pagerank']:
        emb_dims = [0]
    emb_layers = [1, 2]
    if args.emb_type in ['share', 'degree', 'pagerank', 'node2vec', 'deepwalk']:
        emb_layers = [1]
    gnn_layers = [10, 15]
    combinations = [{'lr': lr, 'emb_dim': emb_dim, 'emb_layer': emb_layer, 'gnn_layer': gnn_layer}
                    for lr in lrs for emb_layer in emb_layers for emb_dim in emb_dims for gnn_layer in gnn_layers]

    results = []
    for combination in combinations:
        args.lr = combination['lr']
        args.emb_dim = combination['emb_dim']
        args.emb_layer = combination['emb_layer']
        args.gnn_layer = combination['gnn_layer']

        # prepare node feature
        emb_feats, adj = embedding.node_emb(data=dataset, emb_dim=args.emb_dim, emb_layer=args.emb_layer,
                                            emb_type=args.emb_type, num_subsets=num_subsets)

        train_scores, valid_scores, test_scores, epoch_times = [], [], [], []
        for run in range(args.run_times):
            # set random seed form 0 to 9
            random.seed(run)
            np.random.seed(run)
            torch.cuda.manual_seed(run)
            torch.random.manual_seed(run)

            # prepare model
            args.num_features = emb_feats.shape[-1]
            model = Model(args)

            # start training
            temp_train_score, temp_valid_score, temp_test_score, temp_epoch_time = model.fit(dataset, emb_feats, adj)

            # start testing
            # temp_test_score = model.predict(dataset, emb_feats, adj)
            del model

            train_scores.append(temp_train_score)
            valid_scores.append(temp_valid_score)
            test_scores.append(temp_test_score)
            epoch_times.append(temp_epoch_time)

        train_score_mean = round(np.mean(train_scores), 4)
        train_score_std = round(np.std(train_scores), 4)

        valid_score_mean = round(np.mean(valid_scores), 4)
        valid_score_std = round(np.std(valid_scores), 4)

        test_score_mean = round(np.mean(test_scores), 4)
        test_score_std = round(np.std(test_scores), 4)

        epoch_time_mean = round(np.mean(epoch_times), 4)

        temp = np.array([args.lr, args.weight_decay, args.emb_dim, args.emb_layer, args.gnn_layer, epoch_time_mean,
                         train_score_mean, train_score_std, valid_score_mean, valid_score_std, test_score_mean,
                         test_score_std])
        results.append(temp)

    temp_results = np.array(results, dtype=np.float32)
    best_valid_idx = np.argmax(temp_results[:, -4])
    best_result = temp_results[best_valid_idx, :]
    print(('Mean test Score:{:.4f}, Std test score:{:.4f}').format(best_result[-2], best_result[-1]))

    # record classification results
    np.savetxt(log_dir + '/results.csv', temp_results, fmt='%.05f')
    # record classification results
    records = ['lr', 'weight_decay', 'emb_dim', 'emb_layer', 'gnn_layer', 'epoch_time', 'train_score', 'train_std',
               'valid_score', 'valid_std', 'test_score', 'test_std']
    result_file = open(os.path.join(log_dir, "best_result.txt"), 'w')
    for val in zip(records, best_result):
        result_file.write(val[0] + ':' + np.array2string(val[1]) + '\n')
    result_file.close()


if __name__ == '__main__':
    datasets = ['brazil-airports', 'europe-airports', 'usa-airports']
    types = ['adj', 'EKM', 'MEKL', 'PMEKL', 'SMEKL', 'PSMEKL', 'eigen', 'deepwalk', 'node2vec', 'shared', 'degree', 'pagerank']
    for dataset in datasets:
        for type in types:
            main(dataset, type)