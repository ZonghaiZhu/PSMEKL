# coding:utf-8
import torch, time, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from net import Net
import utils


class Model():
    def __init__(self, args):
        self.args = args
        self.print_freq = args.print_freq
        self.device = args.device
        self.net = Net(args).to(self.device)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        num_params = sum(p.numel() for p in self.net.parameters())
        print(f'#Params: {num_params}')

    def fit(self, dataset, emb_feats, adj):
        best_val = 0
        best_epoch = 0
        val2tst = 0
        self.net.train()

        emb_feats = torch.tensor(emb_feats, dtype=torch.float32).to(self.args.device)
        adj = adj.to(self.args.device)

        train_idx = self.args.train_mask
        val_idx = self.args.val_mask
        test_idx = self.args.test_mask

        train_labels = dataset[0].y.to(self.args.device)

        for epoch in range(1, 1 + self.args.epochs):
            # adjust learning rate
            self.net.train()
            start_time = time.time()
            if epoch > self.args.lr_decay_epoch:
                new_lr = self.args.lr * pow(self.args.lr_decay_rate, (epoch - self.args.lr_decay_epoch))
                new_lr = max(new_lr, 1e-4)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            self.optimizer.zero_grad()
            out = self.net(emb_feats, adj, self.args.emb_type)
            loss = F.nll_loss(out[train_idx], train_labels[train_idx])
            loss.backward()
            self.optimizer.step()

            epoch_time = time.time() - start_time

            pred = out.argmax(dim=1)
            train_acc = accuracy_score(train_labels[train_idx].data.cpu(), pred[train_idx].data.cpu())
            val_acc = accuracy_score(train_labels[val_idx].data.cpu(), pred[val_idx].data.cpu())
            tst_acc = accuracy_score(train_labels[test_idx].data.cpu(), pred[test_idx].data.cpu())

            if val_acc >= best_val:
                best_net = copy.deepcopy(self.net)
                best_val = val_acc
                val2tst = tst_acc
                best_epoch = epoch
            else:
                if epoch >= self.args.least_epoch and epoch - best_epoch > self.args.early_stop:
                    print('\nEarly stop at %d epoch. The best is in %d epoch' %
                          (epoch, best_epoch))
                    self.net = best_net
                    break

            if epoch % self.args.print_freq == 0:
                print(('Epoch:[{}/{}], Epoch_time:{:.3f}\t'
                       'Train_Accuracy:{:.4f}, Train_Loss:{:.4f}\t'
                       'Val_Accuracy:{:.3f}, Tst_Accuracy:{:.3f}').format(
                    epoch + 1, self.args.epochs, epoch_time, train_acc, loss.item(), best_val, val2tst)
                )

            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Training_time", epoch_time)
            utils.log_tabular("Train_Loss", loss.item())
            utils.log_tabular("Train_Acc", train_acc)
            utils.log_tabular("Val_ACC", val_acc)
            utils.log_tabular("Tst_ACC", tst_acc)
            utils.dump_tabular()

        return train_acc, best_val, val2tst, epoch_time

    def predict(self, dataset, emb_feats, adj):
        self.net.eval()

        test_idx = self.args.test_mask
        train_labels = dataset[0].y.to(self.args.device)
        emb_feats = torch.tensor(emb_feats, dtype=torch.float32).to(self.args.device)
        adj = adj.to(self.args.device)

        out = self.net(emb_feats, adj, self.args.emb_type)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(train_labels[test_idx].data.cpu(), pred[test_idx].data.cpu())

        return test_acc