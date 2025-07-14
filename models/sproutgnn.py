# sproutgnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Constants for Dominant policy
USE_PCA = True
PCA_COMPONENTS = 10
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def Dominant(x, edge_index, y):
    x, edge_index, y = x.clone(), edge_index.clone(), y.clone()
    node_features = x.detach().cpu().numpy()

    if USE_PCA and node_features.shape[1] > PCA_COMPONENTS:
        node_features = PCA(n_components=PCA_COMPONENTS).fit_transform(node_features)

    n_clusters = int(torch.unique(y[y >= 0]).numel())
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(node_features)
    centers = kmeans.cluster_centers_[kmeans.labels_]
    distances = np.linalg.norm(node_features - centers, axis=1)

    dist_thr = np.median(distances)
    keep_mask = torch.tensor(distances <= dist_thr, dtype=torch.bool, device=x.device)

    x[~keep_mask] = 0.0

    src, dst = edge_index
    valid_edges = keep_mask[src] & keep_mask[dst]
    edge_index = edge_index[:, valid_edges]

    return x, edge_index

class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=self.lin(x))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return F.relu(aggr_out)

class SproutGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.ego_gnn       = MessagePassingLayer(in_channels, hidden_channels)
        self.cosine_gnn    = MessagePassingLayer(in_channels, hidden_channels)
        self.global_encoder = nn.Linear(in_channels, hidden_channels)
        # final input dim = ego_hidden + dominant_in + cosine_hidden + global_hidden
        self.final_fc = nn.Linear(hidden_channels + in_channels + hidden_channels + hidden_channels,
                                  out_channels)

    def forward(self, x, edge_index, y=None, return_branches=False):
        if y is None:
            raise ValueError("`y` must be provided for the Dominant branch")

        num_nodes = x.size(0)

        # Ego k-hop features
        ego_feats = torch.zeros_like(x, device=x.device)
        for i in range(num_nodes):
            subset, _, _, _ = k_hop_subgraph(i, 2, edge_index,
                                             relabel_nodes=False,
                                             num_nodes=num_nodes)
            if subset.numel():
                ego_feats[i] = x[subset].mean(dim=0)
            else:
                ego_feats[i] = x[i]

        # Dominant-pruned features
        dominant_feats, dominant_ei = Dominant(x, edge_index, y)

        # Cosine-weighted neighbor features
        norm_x = F.normalize(x, p=2, dim=1)
        src, dst = edge_index
        cos_feats = torch.zeros_like(x, device=x.device)
        for i in range(num_nodes):
            nbrs = dst[src == i]
            if nbrs.numel():
                sims = (norm_x[nbrs] * norm_x[i]).sum(dim=1)
                wts = F.softmax(sims, dim=0)
                cos_feats[i] = (x[nbrs] * wts.unsqueeze(1)).sum(dim=0) / wts.sum()
            else:
                cos_feats[i] = x[i]

        # Global linear encoding
        global_feats = self.global_encoder(x)

        # Apply GNNs on ego and cosine branches
        ego_enc    = self.ego_gnn(   ego_feats,       edge_index)
        cosine_enc = self.cosine_gnn(cos_feats,      edge_index)
        # dominant_feats used directly as cut_enc
        cut_enc    = dominant_feats

        combined = torch.cat([ego_enc, cut_enc, cosine_enc, global_feats], dim=-1)
        out = F.log_softmax(self.final_fc(combined), dim=1)

        if return_branches:
            return out, ego_enc, cut_enc, cosine_enc, global_feats
        return out
