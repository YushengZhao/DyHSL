import torch
import torch.nn as nn
import torch.nn.functional as F
from util import norm_adj


class GNNLayer(nn.Module):
    def __init__(self, args, predefined_adj, use_learned_adj=True, padding=0):
        super().__init__()
        self.args = args
        self.layer_norm = nn.LayerNorm(args.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.use_learned_adj = use_learned_adj
        if use_learned_adj:
            self.weights = torch.nn.parameter.Parameter(torch.rand((args.hidden_dim,)))

        adj = torch.zeros((3 * args.num_nodes, 3 * args.num_nodes), requires_grad=False).to(predefined_adj.device)
        adj[:args.num_nodes, :args.num_nodes] = predefined_adj
        adj[args.num_nodes:2*args.num_nodes, args.num_nodes:2*args.num_nodes] = predefined_adj
        adj[-args.num_nodes:, -args.num_nodes:] = predefined_adj
        identity = torch.eye(args.num_nodes).to(adj.device)
        adj = adj + identity.repeat(3, 3)
        self.predefined_adj = norm_adj(adj.unsqueeze(0))  # 1 x 3N x 3N
        self.edge_mask = None
        self.padding = padding

    def forward(self, feat):
        batchsize = feat.size(0)
        feat_dim = feat.size(-1)
        if self.padding > 0:
            pad = torch.zeros(size=(self.args.batch_size, self.padding, self.args.num_nodes, self.args.hidden_dim),
                              device=self.args.device)
            feat = torch.cat([feat, pad], dim=1)

        if self.use_learned_adj:
            weighted_feat = F.normalize(feat * torch.sigmoid(self.weights), p=2, dim=-1)  # B x T x N x D
        else:
            weighted_feat = None

        feat_list = []
        for i in range(2, feat.size(1)):
            feature = feat[:, i-2:i+1, :, :]  # B x 3 x N x D
            feature = feature.reshape((batchsize, -1, feat_dim))  # B x (3 x N) x D
            feature_sum = feat[:, i, :, :]
            if self.use_learned_adj:
                weighted_feature = weighted_feat[:, i-2:i+1, :, :]
                weighted_feature = weighted_feature.reshape((batchsize, -1, feat_dim))
                learned_adj_matrix = weighted_feature @ weighted_feature.transpose(1, 2)  # B x (3 x N) x (3 x N)
                learned_adj_matrix = norm_adj(learned_adj_matrix)
                feature_with_learned_adj = learned_adj_matrix @ feature
                feature_with_learned_adj = self.ffn(feature_with_learned_adj[:, -self.args.num_nodes:, :])
                feature_sum = feature_sum + feature_with_learned_adj
            else:
                if self.edge_mask is None:
                    feature_with_predefined_adj = self.predefined_adj @ feature
                else:
                    feature_with_predefined_adj = norm_adj(self.predefined_adj * self.edge_mask) @ feature
                feature_sum = feature_sum + self.ffn(feature_with_predefined_adj[:, -self.args.num_nodes:, :])
            feature_sum = self.layer_norm(feature_sum)
            feat_list.append(feature_sum)
        new_feat = torch.stack(feat_list, dim=1)  # B x T x N x D
        return new_feat
