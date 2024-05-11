# coding:utf-8
import os, time, argparse
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, Amazon
import torch, random
import utils
from model import Model
import embedding


num_subsets_dict = {'cora': 3, 'citeseer': 3, 'pubmed': 5, 'photo': 3, 'computers': 5}

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--data_path', default='../../DataSet/', type=str, help="data path (dictionary)")
# TUDataset refer to https://chrsmrrs.github.io/datasets/docs/datasets/
parser.add_argument('--dataset', type=str, default="cora",
                    help='cora/citeseer/pubmed/photo/computers')

## For neural network
parser.add_argument('--gnn', type=str, default='gcn', help='gcn, gin,  or sage (default: gcn)')
parser.add_argument('--gnn_layer', type=int, default=15,
                    help='number of GNN message passing layers (default: 15)')
parser.add_argument('--gnn_emb_dim', type=int, default=256,
                    help='number of GNN message passing layers (default: 256)')
parser.add_argument('--emb_layer', type=int, default=1,
                    help='number of layers for node embedding of eigen-based methods (default: 1)')
parser.add_argument('--emb_type', type=str, default='EKM',
                    help='P: EKM, PMEKL, SMEKL, PSMEKL, eigen, eigen_norm, deepwalk, or node2vec | S: shared, degree, pagerank')
parser.add_argument('--emb_dim', type=int, default=128, help='hidden size for node feature')

# For training
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay_epoch', default=10, type=int)
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs')
parser.add_argument('--cls_weight', default=0.5, type=float)
parser.add_argument('--least_epoch', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--early_stop', type=int, default=30, help='patience for early stopping')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument("--run_times", type=int, default=10, help="seed for initializing training.")
parser.add_argument("--folds", type=int, default=10, help="10-folds cross-validation for training.")
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main(type):
    args = parser.parse_args()
    args.emb_type = type
    num_subsets = num_subsets_dict[args.dataset]

    # prepare related data
    data_dict = {'cora': 'planetoid', 'citeseer': 'planetoid', 'pubmed': 'planetoid',
                 'photo': 'amazon', 'computers': 'amazon'} # amazon without train val test index
    target_type = data_dict[args.dataset]
    if target_type == 'planetoid':
        dataset = Planetoid(args.data_path, name=args.dataset)
        args.train_mask = dataset[0]['train_mask']
        args.val_mask = dataset[0]['val_mask']
        args.test_mask = dataset[0]['test_mask']
    elif target_type == 'amazon':
        dataset = Amazon(args.data_path, name=args.dataset)
        args.train_mask, args.val_mask, args.test_mask = [], [], []
        for l in range(dataset.num_classes):
            temp = np.where(dataset[0].y == l)[0]
            args.train_mask.extend(list(temp[:int(0.2 * len(temp))]))
            args.val_mask.extend(list(temp[int(0.2 * len(temp)):int(0.6 * len(temp))]))
            args.test_mask.extend(list(temp[int(0.6 * len(temp)):]))

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    # prepare related documents
    print('\nSetting environment...')

    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = 'log/' + args.dataset + '_' + args.gnn + '_' + args.emb_type + '_' + str(args.dropout) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir)
    utils.configure_output_dir(log_dir)

    # hyperparameters: lr, dropout, weight_decay, emb_dim, permutation
    lrs = [0.001, 0.0001]
    emb_dims = [128, 256]
    if args.dataset == 'pubmed':
        emb_dims = [512]
    if args.emb_type in ['EKM', 'MEKM', 'adj', 'pagerank']:
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
    types = ['adj', 'EKM', 'MEKL', 'PMEKL', 'SMEKL', 'CLMEKL', 'eigen', 'deepwalk', 'node2vec', 'shared', 'degree', 'pagerank']
    for type in types:
        main(type)