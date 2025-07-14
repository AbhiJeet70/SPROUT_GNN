# models/esan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ESAN Model for node-level prediction using subgraph aggregation.
class ESAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ESAN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.shared_aggregator = nn.Linear(hidden_dim, output_dim)

    def forward(self, subgraphs, num_nodes, batch_size=50):
        device = next(self.parameters()).device
        node_predictions = torch.zeros((num_nodes, self.shared_aggregator.out_features), device=device)
        node_counts = torch.zeros(num_nodes, device=device)

        for i in range(0, len(subgraphs), batch_size):
            batch = subgraphs[i:i + batch_size]
            for subgraph in batch:
                x, edge_index = subgraph.x.to(device), subgraph.edge_index.to(device)
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                x = self.shared_aggregator(x)
                node_predictions[subgraph.n_id] += x
                node_counts[subgraph.n_id] += 1

        node_predictions = node_predictions / node_counts.unsqueeze(1).clamp(min=1)
        return F.log_softmax(node_predictions, dim=1)


