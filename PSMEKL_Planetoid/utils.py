# coding:utf-8
import os, json, atexit, time, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}


def configure_output_dir(dir=None):
    G.output_dir = dir
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print("Logging data to %s" % G.output_file.name)


def save_hyperparams(params):
    with open(os.path.join(G.output_dir, "hyperparams.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))


def save_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    torch.save(model, os.path.join(G.output_dir, "model.pkl"))


def load_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    temp = torch.load('model.pkl')
    model.resnet.load_state_dict(temp.resnet.state_dict())
    model.classifier.load_state_dict(temp.classifier.state_dict())


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_tabular(key, val):
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers
    assert key not in G.log_current_row
    G.log_current_row[key] = val


def dump_tabular():
    vals = []
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        vals.append(val)
    if G.output_dir is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False

