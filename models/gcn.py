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
                 use_embs=False,
                 pretrained_embs=None,
                 n_tokens=None,
                 pad_ix=None,
                 dropout=0.5):
        super(GCN, self).__init__()

        self.g = g
        self.use_embs = use_embs
        self.pad_ix = pad_ix
        self.n_tokens = n_tokens

        if use_embs:
            self.emb = nn.Embedding(n_tokens, in_feats, padding_idx=pad_ix)

            if pretrained_embs is not None:
                self.emb.weights = nn.Parameter(pretrained_embs, requires_grad=True)

        self.gcn_layer1 = GraphConv(in_feats, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        if self.use_embs:
            seq_len = torch.sum(features != self.pad_ix, axis=1)

            seq_len = seq_len.view((-1, 1))

            h = self.emb(features)

            h = h.sum(dim=1) / seq_len
        else:
            h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h
