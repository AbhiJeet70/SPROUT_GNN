import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv, GINConv

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='GCN'):
        super(GNN, self).__init__()
        if model_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif model_type == 'GraphSage':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
            self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False)
        elif model_type == 'SGC':
            self.conv1 = SGConv(input_dim, hidden_dim)
            self.conv2 = SGConv(hidden_dim, output_dim)
        elif model_type == 'GIN':
            # For GIN, we use an MLP for each layer.
            self.mlp1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.conv1 = GINConv(self.mlp1)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.conv2 = GINConv(self.mlp2)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
