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

        self.fc1 = nn.Linear(in_feats, 256)
        self.activation = activation
        self.fc2 = nn.Linear(256, n_classes)

        self.gcn_layer1 = GraphConv(in_feats, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def apply_embs(self, features):
        seq_len = torch.sum(features != self.pad_ix, axis=1)

        seq_len = seq_len.view((-1, 1))

        seq_len[seq_len == 0] = 1

        h = self.emb(features)

        h = h.max(dim=1)[0]

        return h

    def forward(self, features):
        if self.use_embs:
            h = self.apply_embs(features)

        else:
            h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h

    def forward_text(self, features):
        h = self.apply_embs(features)

        h = self.activation(self.fc1(h))

        h = self.fc2(h)

        return h

    def freeze_features(self, freeze):
        self.emb.weight.requires_grad = not freeze

    def freeze_graph(self, freeze):
        self.gcn_layer1.weight.requires_grad = not freeze
        self.gcn_layer2.weight.requires_grad = not freeze


class GCN_LSTM(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 use_embs=False,
                 pretrained_embs=None,
                 lstm_num_layers=1,
                 n_tokens=None,
                 pad_ix=None,
                 dropout=0.5):
        super(GCN_LSTM, self).__init__()

        self.g = g
        self.use_embs = use_embs
        self.pad_ix = pad_ix
        self.n_tokens = n_tokens

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = 64

        if use_embs:
            self.emb = nn.Embedding(n_tokens, in_feats, padding_idx=pad_ix)

            if pretrained_embs is not None:
                self.emb.weights = nn.Parameter(pretrained_embs, requires_grad=True)

            self.lstm = nn.LSTM(in_feats, self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=False)

            conv_inp = self.lstm_hidden_size
        else:
            conv_inp = in_feats

        self.fc1 = nn.Linear(conv_inp, 256)
        self.activation = activation
        self.fc2 = nn.Linear(256, n_classes)

        self.gcn_layer1 = GraphConv(conv_inp, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

        self.graph_freeze = False
        self.text_freeze = False

    def forward(self, features):
        if self.graph_freeze:
            return self.forward_text(features)

        if self.use_embs:
            h = self.emb(features)

            h = h.permute(1, 0, 2)

            h_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))
            c_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))

            if torch.cuda.is_available():
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            output, (final_hidden_state, final_cell_state) = self.lstm(h, (h_0, c_0))

            h = final_hidden_state[-1]
        else:
            h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h

    def forward_text(self, features):
        h = self.emb(features)

        h = h.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))
        c_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))

        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        output, (final_hidden_state, final_cell_state) = self.lstm(h, (h_0, c_0))

        h = final_hidden_state[-1]

        h = self.dropout(self.activation(self.fc1(h)))

        h = self.fc2(h)

        return h

    def freeze_features(self, freeze):
        self.text_freeze = freeze
        self.emb.weight.requires_grad = not freeze
        self.fc1.weight.requires_grad = not freeze
        self.fc2.weight.requires_grad = not freeze

    def freeze_graph(self, freeze):
        self.graph_freeze = freeze
        self.gcn_layer1.weight.requires_grad = not freeze
        self.gcn_layer2.weight.requires_grad = not freeze



class GCN_CNN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 use_embs=False,
                 pretrained_embs=None,
                 lstm_num_layers=2,
                 n_tokens=None,
                 pad_ix=None,
                 dropout=0.5):
        super(GCN_CNN, self).__init__()

        self.g = g
        self.use_embs = use_embs
        self.pad_ix = pad_ix
        self.n_tokens = n_tokens

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = 64

        if use_embs:
            self.emb = nn.Embedding(n_tokens, in_feats, padding_idx=pad_ix)

            if pretrained_embs is not None:
                self.emb.weights = nn.Parameter(pretrained_embs, requires_grad=True)

            self.lstm = nn.LSTM(in_feats, self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=False)

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

            h_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))
            c_0 = Variable(torch.zeros(1 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))

            if torch.cuda.is_available():
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            output, (final_hidden_state, final_cell_state) = self.lstm(h, (h_0, c_0))

            h = final_hidden_state[-1]
        else:
            h = features

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h


class GCN_Attention(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 use_embs=False,
                 pretrained_embs=None,
                 lstm_num_layers=2,
                 n_tokens=None,
                 pad_ix=None,
                 dropout=0.5):
        super(GCN_Attention, self).__init__()

        self.g = g
        self.use_embs = use_embs
        self.pad_ix = pad_ix
        self.n_tokens = n_tokens

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = 64

        self.emb = nn.Embedding(n_tokens, in_feats, padding_idx=pad_ix)

        if pretrained_embs is not None:
            self.emb.weights = nn.Parameter(pretrained_embs, requires_grad=True)

        self.lstm = nn.LSTM(in_feats, self.lstm_hidden_size, num_layers=self.lstm_num_layers, bidirectional=True,
                            dropout=0.2)
        self.W_s1 = nn.Linear(2 * self.lstm_hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)

        self.gcn_layer1 = GraphConv(30*2*self.lstm_hidden_size, n_hidden, activation=activation)

        self.gcn_layer2 = GraphConv(n_hidden, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.dropout(self.W_s2(F.tanh(self.W_s1(lstm_output))))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, features):
        # seq_len = torch.sum(features != self.pad_ix, axis=1)

        # seq_len = seq_len.view((-1, 1))

        h = self.emb(features)

        # h = h.sum(dim=1) / seq_len

        h = h.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(2 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))
        c_0 = Variable(torch.zeros(2 * self.lstm_num_layers, h.shape[1], self.lstm_hidden_size))

        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        output, (final_hidden_state, final_cell_state) = self.lstm(h, (h_0, c_0))

        output = output.permute(1, 0, 2)

        attn_weight_matrix = self.attention_net(output)

        hidden_matrix = torch.bmm(attn_weight_matrix, output)

        h = hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])

        h = self.gcn_layer1(self.g, h)

        h = self.dropout(h)

        h = self.gcn_layer2(self.g, h)

        return h
