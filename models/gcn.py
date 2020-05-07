"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout):
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
