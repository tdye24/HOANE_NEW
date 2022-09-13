import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from layers import GCN, GAT, DenseModel, NodeAttrAttention
from utils import sample_n


class HOANE(nn.Module):
    """
    Node Encoder: GAT
    Attr Encoder: MLP
    Node Decoder: Inner Product
    Attr Decoder:
    """

    def __init__(self, num_nodes=2708, input_dim=1433, num_hidden=128, out_dim=512, noise_dim=5,
                 K=1, J=1, dropout=0., device=None,
                 encoder_type='gcn',
                 encoder_layers=2,
                 decoder_type='gcn',
                 decoder_layers=2,
                 heads=4,
                 node_attr_attention=False,
                 node_attr_attention_dropout=0.0):
        super(HOANE, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.node_attr_attention = node_attr_attention

        if self.encoder_type == 'gcn':
            self.node_mu_nn = GCN(in_dim=input_dim + noise_dim, hid_dim=num_hidden, out_dim=out_dim,
                                  n_layers=encoder_layers,
                                  dropout=dropout)
            self.node_var_nn = GCN(in_dim=input_dim, hid_dim=num_hidden, out_dim=out_dim, n_layers=encoder_layers,
                                   dropout=dropout)
        else:
            assert self.encoder_type == 'gat'
            # assert heads * out_dim == 512
            self.node_mu_nn = GAT(in_dim=input_dim + noise_dim,
                                  hid_dim=num_hidden,
                                  out_dim=out_dim,
                                  n_layers=encoder_layers,
                                  heads=heads,
                                  dropout=dropout,
                                  alpha=0.2)
            self.node_var_nn = GAT(in_dim=input_dim,
                                   hid_dim=num_hidden,
                                   out_dim=out_dim,
                                   n_layers=encoder_layers,
                                   heads=heads,
                                   dropout=dropout,
                                   alpha=0.2)

        self.attr_mu_nn = DenseModel(in_dim=num_nodes + noise_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                     dropout=dropout)
        self.attr_var_nn = DenseModel(in_dim=num_nodes, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                      dropout=dropout)

        if self.decoder_type == 'gcn':
            self.decoder = GCN(in_dim=2 * out_dim,
                               hid_dim=num_hidden,
                               out_dim=input_dim,
                               n_layers=decoder_layers,
                               dropout=dropout)

        if self.decoder_type == 'gat':
            self.decoder = GAT(in_dim=2 * out_dim,
                               hid_dim=num_hidden,
                               out_dim=input_dim,
                               n_layers=decoder_layers,
                               heads=heads,
                               dropout=dropout,
                               alpha=0.2)
        if self.decoder_type in ['gcn', 'gat'] and node_attr_attention:
            self.node_attr_attention_layer = NodeAttrAttention(in_features=out_dim,
                                                               out_features=out_dim,
                                                               dropout=node_attr_attention_dropout,
                                                               alpha=0.2)
        self.noise_dim = noise_dim
        self.noise_dist = dist.Bernoulli(torch.tensor([.5], device=device))

        self.K = K
        self.J = J

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'attr_mu_nn' in name or 'attr_var_nn' in name:
                # print(name)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    assert 'bias' in name
                    torch.nn.init.zeros_(param)

    def encode(self, adj, x):
        attr_dim = x.shape[1]
        num_nodes = x.shape[0]
        node_mu_input = x.unsqueeze(1).repeat(1, self.K + self.J, 1)
        node_noise_e = self.noise_dist.sample(torch.Size([num_nodes, self.K + self.J, self.noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.cat((node_noise_e, node_mu_input), 2)
        node_mu = self.node_mu_nn(adj, noised_node_mu_input)
        node_mu_iw = node_mu[:, :self.K, :]
        node_mu_star = node_mu[:, self.K:, :]
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)

        node_logv = self.node_var_nn(adj, x)
        node_logv_iw = node_logv.unsqueeze(1).repeat(1, self.K, 1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        node_mu_star1 = node_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_node_mu = torch.cat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)

        attr_mu_input = x.transpose(0, 1).unsqueeze(1).repeat(1, self.K + self.J, 1)
        attr_noise_e = self.noise_dist.sample(torch.Size([attr_dim, self.K + self.J, self.noise_dim]))
        attr_noise_e = torch.squeeze(attr_noise_e)
        noised_attr_mu_input = torch.cat((attr_noise_e, attr_mu_input), 2)
        attr_mu = self.attr_mu_nn(noised_attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]
        attr_mu_star = attr_mu[:, self.K:, :]
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)

        attr_logv = self.attr_var_nn(x=x.transpose(0, 1))
        attr_logv_iw = attr_logv.unsqueeze(1).repeat(1, self.K, 1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        attr_mu_star1 = attr_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_attr_mu = torch.cat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec

    def decode(self, node_z, attr_z, adj, x):
        outputs_u, outputs_a = None, None
        for i in range(self.K):
            input_u = node_z[:, i, :].squeeze()
            input_a = attr_z[:, i, :].squeeze()

            z_u = F.dropout(input_u, self.dropout, self.training)
            z_u_t = z_u.transpose(0, 1)
            logits_node = torch.matmul(z_u, z_u_t)

            if self.decoder_type == 'inner_product':
                z_a = F.dropout(input_a, self.dropout, self.training)
                z_a_t = z_a.transpose(0, 1)
                logits_attr = torch.matmul(z_u, z_a_t)
            else:
                assert self.decoder_type in ['gcn', 'gat']
                z_a = F.dropout(input_a, self.dropout, self.training)
                if self.node_attr_attention:
                    fine_grained_features = self.node_attr_attention_layer(node_embedding=z_u,
                                                                           attr_embedding=z_a,
                                                                           feat_mat=x)
                else:
                    weights = F.normalize(x, p=1, dim=1)
                    fine_grained_features = torch.matmul(weights, z_a)
                concat_features = torch.cat((z_u, fine_grained_features), dim=1)
                logits_attr = self.decoder(adj, concat_features)

            if i == 0:
                outputs_u = logits_node.unsqueeze(2)
                outputs_a = logits_attr.unsqueeze(2)
            else:
                outputs_u = torch.cat((outputs_u, logits_node.unsqueeze(2)), 2)
                outputs_a = torch.cat((outputs_a, logits_attr.unsqueeze(2)), 2)
        return outputs_u, outputs_a

    def forward(self, adj, x):
        merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
        merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
        node_mu_iw_vec, attr_mu_iw_vec = self.encode(adj, x)
        assert node_mu_iw_vec.shape[1] == 512
        assert attr_mu_iw_vec.shape[1] == 512
        reconstruct_node_logits, reconstruct_attr_logits = self.decode(adj=adj,
                                                                       x=x,
                                                                       node_z=node_z_samples_iw,
                                                                       attr_z=attr_z_samples_iw)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               reconstruct_node_logits, reconstruct_attr_logits, node_mu_iw_vec, attr_mu_iw_vec


class HOANE_V2(HOANE):
    def __init__(self, num_nodes=2708, input_dim=1433, num_hidden=128, out_dim=512, noise_dim=5,
                 K=1, J=1, dropout=0., device=None,
                 encoder_type='gcn',
                 encoder_layers=2,
                 decoder_type='gcn',
                 decoder_layers=2,
                 heads=4,
                 node_attr_attention=False,
                 node_attr_attention_dropout=0.0,
                 encoder_node_attr_attention=False,
                 encoder_node_attr_attention_dropout=0.0):
        super(HOANE_V2, self).__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            noise_dim=noise_dim,
            K=K,
            J=J,
            dropout=dropout,
            device=device,
            encoder_type=encoder_type,
            encoder_layers=encoder_layers,
            decoder_type=decoder_type,
            decoder_layers=decoder_layers,
            heads=heads,
            node_attr_attention=node_attr_attention,
            node_attr_attention_dropout=node_attr_attention_dropout
        )
        self.attr_mu_nn = DenseModel(in_dim=out_dim + noise_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                     dropout=dropout)
        self.attr_var_nn = DenseModel(in_dim=out_dim, num_hidden=num_hidden, out_dim=out_dim, num_layers=2,
                                      dropout=dropout)
        self.encoder_node_attr_attention = encoder_node_attr_attention

        if self.encoder_type in ['gcn', 'gat'] and encoder_node_attr_attention:
            self.encoder_node_attr_attention_layer = NodeAttrAttention(in_features=out_dim,
                                                                       out_features=out_dim,
                                                                       dropout=encoder_node_attr_attention_dropout,
                                                                       alpha=0.2)

    def encode(self, adj, x):
        attr_dim = x.shape[1]
        num_nodes = x.shape[0]
        node_mu_input = x.unsqueeze(1).repeat(1, self.K + self.J, 1)
        node_noise_e = self.noise_dist.sample(torch.Size([num_nodes, self.K + self.J, self.noise_dim]))
        node_noise_e = torch.squeeze(node_noise_e)
        noised_node_mu_input = torch.cat((node_noise_e, node_mu_input), 2)
        node_mu = self.node_mu_nn(adj, noised_node_mu_input)
        node_mu_iw = node_mu[:, :self.K, :]
        node_mu_star = node_mu[:, self.K:, :]
        node_mu_iw_vec = torch.mean(node_mu_iw, 1)

        node_logv = self.node_var_nn(adj, x)
        node_logv_iw = node_logv.unsqueeze(1).repeat(1, self.K, 1)
        node_sigma_iw1 = torch.exp(0.5 * node_logv_iw)
        merged_node_sigma = node_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        node_z_samples_iw = sample_n(mu=node_mu_iw, sigma=node_sigma_iw1)
        merged_node_z_samples = node_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        node_mu_star1 = node_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_node_mu = torch.cat((node_mu_star1, node_mu_iw.unsqueeze(2)), 2)

        # generate mu of attrs conditioning on mu of nodes
        if self.encoder_node_attr_attention:
            pass
        else:
            weights = F.normalize(x.transpose(0, 1), p=1, dim=1).unsqueeze(1).repeat(1, self.K + self.J, 1)
            attr_mu_input = torch.matmul(weights.transpose(0, 1), node_mu.transpose(0, 1)).transpose(0, 1)
            attr_noise_e = self.noise_dist.sample(torch.Size([attr_dim, self.K + self.J, self.noise_dim]))
            attr_noise_e = torch.squeeze(attr_noise_e)
            noised_attr_mu_input = torch.cat((attr_noise_e, attr_mu_input), 2)  # 1433x2x(512+5)

        attr_mu = self.attr_mu_nn(noised_attr_mu_input)
        attr_mu_iw = attr_mu[:, :self.K, :]
        attr_mu_star = attr_mu[:, self.K:, :]
        attr_mu_iw_vec = torch.mean(attr_mu_iw, 1)

        # generate logv of attrs conditioning on logv of nodes
        if self.encoder_node_attr_attention:
            pass
        else:
            weights = F.normalize(x.transpose(0, 1), p=1, dim=1)
            attr_logv_input = torch.matmul(weights, node_logv)
        attr_logv = self.attr_var_nn(x=attr_logv_input)
        attr_logv_iw = attr_logv.unsqueeze(1).repeat(1, self.K, 1)
        attr_sigma_iw1 = torch.exp(0.5 * attr_logv_iw)
        merged_attr_sigma = attr_sigma_iw1.unsqueeze(2).repeat(1, 1, self.J + 1, 1)

        attr_z_samples_iw = sample_n(mu=attr_mu_iw, sigma=attr_sigma_iw1)
        merged_attr_z_samples = attr_z_samples_iw.unsqueeze(2).repeat(1, 1, self.J + 1, 1)
        attr_mu_star1 = attr_mu_star.unsqueeze(1).repeat(1, self.K, 1, 1)
        merged_attr_mu = torch.cat((attr_mu_star1, attr_mu_iw.unsqueeze(2)), 2)

        return merged_node_mu, merged_node_sigma, merged_node_z_samples, node_logv_iw, node_z_samples_iw, \
               merged_attr_mu, merged_attr_sigma, merged_attr_z_samples, attr_logv_iw, attr_z_samples_iw, \
               node_mu_iw_vec, attr_mu_iw_vec
