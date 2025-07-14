# models/sagn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
import networkx as nx

# Enable cuDNN benchmarking for improved performance when input sizes are fixed.
torch.backends.cudnn.benchmark = True

class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MessagePassingLayer, self).__init__(aggr="add")
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply the linear transformation on GPU and propagate messages.
        return self.propagate(edge_index, x=self.linear(x))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return F.relu(aggr_out)

class SubstructureAwareGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SubstructureAwareGNN, self).__init__()
        self.ego_gnn = MessagePassingLayer(in_channels, hidden_channels)
        self.cut_gnn = MessagePassingLayer(in_channels, hidden_channels)
        self.global_encoder = nn.Linear(in_channels, hidden_channels)
        self.final_fc = nn.Linear(3 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Extract subgraph features
        ego_features = self.extract_ego_subgraph(x, edge_index)
        cut_features = self.extract_cut_subgraph(x, edge_index)
        # Process features through respective GNN layers
        ego_encoded = self.ego_gnn(ego_features, edge_index)
        cut_encoded = self.cut_gnn(cut_features, edge_index)
        global_encoded = self.global_encoder(x)
        # Concatenate and classify
        combined_features = torch.cat([ego_encoded, cut_encoded, global_encoded], dim=-1)
        output = self.final_fc(combined_features)
        return F.log_softmax(output, dim=1)

    def extract_ego_subgraph(self, x, edge_index):
        """
        For each node, compute the mean feature vector over its k-hop subgraph (k=2).
        Note: This loop is executed sequentially and on CPU if not vectorized.
        For better GPU utilization, consider refactoring to a batched operation.
        """
        k = 2
        num_nodes = x.size(0)
        ego_features = torch.zeros_like(x, device=x.device)
        for node_idx in range(num_nodes):
            # k_hop_subgraph returns (subset, edge_index, mapping, mask)
            subset, _, _, _ = k_hop_subgraph(node_idx, k, edge_index, relabel_nodes=False, num_nodes=num_nodes)
            if subset.numel() > 0:
                # x[subset] is on the GPU since x is on x.device.
                ego_features[node_idx] = x[subset].mean(dim=0)
            else:
                ego_features[node_idx] = x[node_idx]
        return ego_features

    def extract_cut_subgraph(self, x, edge_index):
        """
        Remove half of the edges (selected randomly) and compute mean feature over the remaining neighbors.
        """
        num_edges = edge_index.size(1)
        num_remove = num_edges // 2
        # Select random indices to remove (this runs on GPU if edge_index is on GPU)
        indices = torch.randperm(num_edges, device=x.device)[:num_remove]
        mask = torch.ones(num_edges, dtype=torch.bool, device=x.device)
        mask[indices] = False
        new_edge_index = edge_index[:, mask]
        num_nodes = x.size(0)
        cut_features = torch.zeros_like(x, device=x.device)
        for node_idx in range(num_nodes):
            neighbors = new_edge_index[1][new_edge_index[0] == node_idx]
            if neighbors.numel() > 0:
                cut_features[node_idx] = x[neighbors].mean(dim=0)
            else:
                cut_features[node_idx] = x[node_idx]
        return cut_features
