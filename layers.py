import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        torch.nn.init.uniform_(self.weight, a=-init_range, b=init_range)

    def forward(self, adj, x):
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layers, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # init_range = np.sqrt(6.0 / (self.in_features + self.out_features))
        # torch.nn.init.uniform_(self.W, a=-init_range, b=init_range)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, h):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(p=dropout)
        self.convs = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GraphConvolution(in_dim, out_dim))
        else:
            self.convs.append(GraphConvolution(in_dim, hid_dim))
            for i in range(n_layers - 2):
                self.convs.append(GraphConvolution(hid_dim, hid_dim))
            self.convs.append(GraphConvolution(hid_dim, out_dim))

    def forward(self, g, inputs):
        outputs = None
        if len(inputs.shape) == 2:  # GCN
            h = inputs
            for l in range(self.n_layers - 1):
                h = self.dropout_layer(h)
                h = F.relu(self.convs[l](g, h))
            h = self.dropout_layer(h)
            h = self.convs[-1](g, h)
            outputs = h
        else:
            assert len(inputs.shape) == 3
            K = inputs.shape[1]
            for i in range(K):
                h = inputs[:, i, :].squeeze(1)
                for l in range(self.n_layers - 1):
                    h = self.dropout_layer(h)
                    h = F.relu(self.convs[l](g, h))
                h = self.dropout_layer(h)
                h = self.convs[-1](g, h)
                if i == 0:
                    outputs = h.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, h.unsqueeze(1)), dim=1)
        return outputs


class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout, alpha, heads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.num_layers = n_layers
        self.heads = heads
        self.gat_layers = []
        if n_layers == 1:
            # 一层Graph attention layer
            self.out_att = GraphAttentionLayer(in_features=in_dim,
                                               out_features=out_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)
        else:
            self.gat_layers.extend(
                [GraphAttentionLayer(in_features=in_dim,
                                     out_features=hid_dim,
                                     dropout=dropout,
                                     alpha=alpha,
                                     concat=True) for _ in range(heads)]
            )
            for i in range(1, n_layers - 1):
                self.gat_layers.extend(
                    [GraphAttentionLayer(in_features=hid_dim * heads,
                                         out_features=hid_dim,
                                         dropout=dropout,
                                         alpha=alpha,
                                         concat=True) for _ in range(heads)]
                )
            self.gat_layers = nn.ModuleList(self.gat_layers)
            self.out_att = GraphAttentionLayer(in_features=hid_dim * heads,
                                               out_features=out_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)

    def forward(self, g, inputs):
        outputs = None
        if len(inputs.shape) == 2:  # GCN
            h = inputs
            for i in range(0, self.num_layers - 1):
                h = F.dropout(h, self.dropout, training=self.training)
                h = torch.cat([att(g, h) for att in self.gat_layers[i * self.heads: (i + 1) * self.heads]], dim=1)
            h = F.dropout(h, self.dropout, training=self.training)
            outputs = self.out_att(g, h)
        else:
            assert len(inputs.shape) == 3
            K = inputs.shape[1]
            for i in range(K):
                h = inputs[:, i, :].squeeze(1)
                for l in range(0, self.num_layers - 1):
                    h = F.dropout(h, self.dropout, training=self.training)
                    h = torch.cat([att(g, h) for att in self.gat_layers[l * self.heads: (l + 1) * self.heads]], dim=1)
                h = F.dropout(h, self.dropout, training=self.training)
                h = self.out_att(g, h)
                if i == 0:
                    outputs = h.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, h.unsqueeze(1)), dim=1)
        return outputs


class DenseModel(nn.Module):
    """Stack of fully connected layers."""

    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout):
        super(DenseModel, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(p=dropout)

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, num_hidden))
            for l in range(1, num_layers - 1):
                self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.Linear(num_hidden, out_dim))

    def forward(self, x):
        for l in range(self.num_layers - 1):
            x = self.dropout_layer(x)
            x = self.layers[l](x)
            x = torch.tanh(x)

        x = self.dropout_layer(x)
        x = self.layers[-1](x)

        return x


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        x = self.dropout_layer(x)
        logits = self.linear(x)
        return logits


class NodeAttrAttention(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(NodeAttrAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node_embedding, attr_embedding, feat_mat):
        node_h = torch.mm(node_embedding, self.W)
        attr_h = torch.mm(attr_embedding, self.W)

        e = self._prepare_attentional_mechanism_input(node_h, attr_h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(feat_mat > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, attr_h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, node_h, attr_h):
        Wh1 = torch.matmul(node_h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(attr_h, self.a[self.out_features:, :])

        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakrelu(e)
