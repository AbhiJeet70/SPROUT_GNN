# models/sun.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, k_hop_subgraph

class SUNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SUNLayer, self).__init__()
        # Separate transformations for root and non-root nodes
        self.root_mlp = nn.Linear(in_channels, out_channels)
        self.non_root_mlp = nn.Linear(in_channels, out_channels)
        self.global_mlp = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, subgraph_masks):
        # Convert edge_index to dense adjacency matrix
        adjacency_matrix = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
        # Local message passing within subgraphs
        local_features = torch.matmul(adjacency_matrix, x)
        # Global aggregation across subgraphs
        global_features = self.global_mlp(torch.mean(x, dim=0, keepdim=True))
        global_features = global_features.expand(x.size(0), global_features.size(1))
        # Initialize root and non-root features
        root_features = torch.zeros((x.size(0), global_features.size(1)), device=x.device)
        non_root_features = torch.zeros((x.size(0), global_features.size(1)), device=x.device)
        # Apply transformation to root nodes
        root_features[subgraph_masks] = self.root_mlp(x[subgraph_masks])
        # Apply transformation to non-root nodes
        non_root_features = self.non_root_mlp(local_features)
        # Combine features
        updated_features = root_features + non_root_features + global_features
        return updated_features

class SUN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super(SUN, self).__init__()
        self.layer1 = SUNLayer(num_features, hidden_channels)
        self.layer2 = SUNLayer(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        subgraph_masks = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        subgraph_masks[:] = True
        x = self.layer1(x, edge_index, subgraph_masks)
        x = F.relu(x)
        x = self.layer2(x, edge_index, subgraph_masks)
        return F.log_softmax(x, dim=1)
