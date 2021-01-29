import time

import torch
import torch.nn.functional as F
import numpy as np
import dgl
from dgl import DGLGraph


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from datasets import Cora, CiteseerM10, Dblp
from text_transformers import TFIDF, Index, BOW


"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout=0.5):
        super(GCN, self).__init__()

        self.g = g

        self.gcn_layer1 = GraphConv(in_feats, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h

    def freeze_features(self, freeze):
        self.emb.weight.requires_grad = not freeze

    def freeze_graph(self, freeze):
        self.gcn_layer1.weight.requires_grad = not freeze
        self.gcn_layer2.weight.requires_grad = not freeze


def get_masks(n,
              main_ids,
              main_labels,
              test_ratio,
              val_ratio,
              seed=1):
    train_mask = np.zeros(n)
    val_mask = np.zeros(n)
    test_mask = np.zeros(n)

    x_dev, x_test, y_dev, y_test = train_test_split(main_ids,
                                                    main_labels,
                                                    stratify=main_labels,
                                                    test_size=test_ratio,
                                                    random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(x_dev,
                                                      y_dev,
                                                      stratify=y_dev,
                                                      test_size=val_ratio,
                                                      random_state=seed)

    train_mask[x_train] = 1
    val_mask[x_val] = 1
    test_mask[x_test] = 1

    return train_mask, val_mask, test_mask


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask].detach().cpu().numpy()
        _, predicted = torch.max(logits, dim=1)
        predicted = predicted.detach().cpu().numpy()
        f1 = f1_score(labels, predicted, average='micro')
        return f1


def train_gcn(dataset,
              test_ratio=0.5,
              val_ratio=0.2,
              seed=1,
              n_hidden=64,
              n_epochs=200,
              lr=1e-2,
              weight_decay=5e-4,
              dropout=0.5,
              verbose=True,
              cuda=False):
    data = dataset.get_data()

    features = torch.FloatTensor(data['features'])
    labels = torch.LongTensor(data['labels'])

    n = len(data['ids'])
    train_mask, val_mask, test_mask = get_masks(n,
                                                data['main_ids'],
                                                data['main_labels'],
                                                test_ratio=test_ratio,
                                                val_ratio=val_ratio,
                                                seed=seed)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    if cuda:
        torch.cuda.set_device("cuda:0")
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = DGLGraph(data['graph'])
    g = dgl.transform.add_self_loop(g)
    n_edges = g.number_of_edges()

    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        norm = norm.cuda()

    g.ndata['norm'] = norm.unsqueeze(1)

    in_feats = features.shape[1]

    # + 1 for unknown class
    n_classes = data['n_classes'] + 1
    model = GCN(g,
                in_feats=in_feats,
                n_hidden=n_hidden,
                n_classes=n_classes,
                activation=F.relu,
                dropout=dropout)
    if cuda:
        model.cuda()

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, min_lr=1e-10)

    best_f1 = -100
    # initialize graph
    dur = []
    for epoch in range(n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        f1 = evaluate(model, features, labels, val_mask)
        scheduler.step(1 - f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_model.pt')

        if verbose:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | F1 {:.4f} | "
                  "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                f1, n_edges / np.mean(dur) / 1000))

    model.load_state_dict(torch.load('best_model.pt'))
    f1 = evaluate(model, features, labels, test_mask)

    if verbose:
        print()
        print("Test F1 {:.2}".format(f1))

    return f1


def main():
    dataset = Cora()
    # transformer = Index()
    transformer = TFIDF()
    dataset.transform_features(transformer)
    # dataset.features = np.random.rand(len(dataset.features), 100)
    train_gcn(dataset, lr=1e-2, n_epochs=200, verbose=True)


if __name__ == '__main__':
    main()
