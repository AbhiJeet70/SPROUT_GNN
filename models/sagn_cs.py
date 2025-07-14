# models/sagn_cs.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
import networkx as nx

# Enable cuDNN benchmarking for potentially faster performance when input sizes are constant.
torch.backends.cudnn.benchmark = True

class SubstructureAwareGNN_CS(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SubstructureAwareGNN_CS, self).__init__()
        self.ego_gnn = MessagePassingLayer(in_channels, hidden_channels)
        self.cut_gnn = MessagePassingLayer(in_channels, hidden_channels)
        self.cosine_gnn = MessagePassingLayer(in_channels, hidden_channels)
        self.global_encoder = nn.Linear(in_channels, hidden_channels)
        self.final_fc = nn.Linear(4 * hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, return_branches=False):
        # Extract subgraph features from three branches:
        ego_features = self.extract_ego_subgraph(x, edge_index)
        cut_features = self.extract_cut_subgraph(x, edge_index, edge_weight)
        cosine_features = self.extract_cosine_subgraph(x, edge_index)
        
        # Process each branch with its own GNN layer.
        ego_encoded = self.ego_gnn(ego_features, edge_index)
        cut_encoded = self.cut_gnn(cut_features, edge_index)
        cosine_encoded = self.cosine_gnn(cosine_features, edge_index)
        
        # Encode global node features.
        global_encoded = self.global_encoder(x)
        
        # Concatenate features from all branches and classify.
        combined_features = torch.cat([ego_encoded, cut_encoded, cosine_encoded, global_encoded], dim=-1)
        log_probs = F.log_softmax(self.final_fc(combined_features), dim=1)
        
        if return_branches:
            return log_probs, ego_encoded, cut_encoded, global_encoded, cosine_encoded
        return log_probs

    def extract_ego_subgraph(self, x, edge_index):
        # k-hop subgraph extraction: This loop is executed on CPU.
        # For full GPU utilization, consider vectorizing this process.
        k = 2
        num_nodes = x.size(0)
        # Allocate tensor directly on GPU.
        ego_features = torch.zeros_like(x, device=x.device)
        for i in range(num_nodes):
            subset, _, _, _ = k_hop_subgraph(i, k, edge_index, relabel_nodes=False)
            if subset.numel() > 0:
                # Mean over subgraph nodes; this operation is on GPU if x is on GPU.
                ego_features[i] = x[subset].mean(dim=0)
            else:
                ego_features[i] = x[i]
        return ego_features

    def extract_cut_subgraph(self, x, edge_index, edge_weight=None):
        # If no edge weights are provided, assume all are 1.
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        num_nodes = x.size(0)
        cut_features = torch.zeros_like(x, device=x.device)
        src, dst = edge_index[0], edge_index[1]
        for i in range(num_nodes):
            mask = (src == i)
            neighbors = dst[mask]
            if neighbors.numel() > 0:
                weights = edge_weight[mask]
                neighbor_feats = x[neighbors]
                weighted_sum = (neighbor_feats * weights.unsqueeze(1)).sum(dim=0)
                total_weight = weights.sum()
                mean_val = weighted_sum / total_weight if total_weight > 0 else x[i]
                cut_features[i] = mean_val
            else:
                cut_features[i] = x[i]
        return cut_features

    def extract_cosine_subgraph(self, x, edge_index):
        # Normalize features on GPU.
        norm_x = F.normalize(x, p=2, dim=1)
        num_nodes = x.size(0)
        cosine_features = torch.zeros_like(x, device=x.device)
        src, dst = edge_index[0], edge_index[1]
        for i in range(num_nodes):
            mask = (src == i)
            neighbor_indices = dst[mask]
            if neighbor_indices.numel() > 0:
                neighbor_norm = norm_x[neighbor_indices]
                node_norm = norm_x[i].unsqueeze(0)
                cos_sim = (neighbor_norm * node_norm).sum(dim=1)
                weights = F.softmax(cos_sim, dim=0)
                neighbor_feats = x[neighbor_indices]
                weighted_sum = (neighbor_feats * weights.unsqueeze(1)).sum(dim=0)
                total_weight = weights.sum()
                mean_val = weighted_sum / total_weight if total_weight > 0 else x[i]
                cosine_features[i] = mean_val
            else:
                cosine_features[i] = x[i]
        return cosine_features

class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MessagePassingLayer, self).__init__(aggr="add")
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Ensure the input is processed by the linear layer on GPU.
        return self.propagate(edge_index, x=self.linear(x))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return F.relu(aggr_out)

