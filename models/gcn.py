"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
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

        self.lstm_hidden_size = 64

        if use_embs:
            self.emb = nn.Embedding(n_tokens, in_feats, padding_idx=pad_ix)

            if pretrained_embs is not None:
                self.emb.weights = nn.Parameter(pretrained_embs, requires_grad=True)

            self.lstm = nn.LSTM(in_feats, self.lstm_hidden_size)

            conv_inp = self.lstm_hidden_size
        else:
            conv_inp = in_feats

        self.gcn_layer1 = GraphConv(conv_inp, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        if self.use_embs:
            # seq_len = torch.sum(features != self.pad_ix, axis=1)

            # seq_len = seq_len.view((-1, 1))

            h = self.emb(features)

            # h = h.sum(dim=1) / seq_len

            h = h.permute(1, 0, 2)

            h_0 = Variable(torch.zeros(1, h.shape[1], self.lstm_hidden_size))
            c_0 = Variable(torch.zeros(1, h.shape[1], self.lstm_hidden_size))

            output, (final_hidden_state, final_cell_state) = self.lstm(h, (h_0, c_0))

            h = final_hidden_state[-1]
        else:
            h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h
