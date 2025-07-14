# dataset.py
import torch
import numpy as np
import random
import networkx as nx
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()

def load_dataset(dataset_name):
    if dataset_name in ["Cora", "PubMed", "CiteSeer"]:
        dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    elif dataset_name == "Flickr":
        dataset = Flickr(root="./data/Flickr")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset

def split_dataset(data, test_size=0.2, val_size=0.1):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    num_test = int(test_size * num_nodes)
    num_val = int(val_size * num_nodes)
    num_train = num_nodes - num_test - num_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    data.train_mask = train_mask.to(data.x.device)
    data.val_mask = val_mask.to(data.x.device)
    data.test_mask = test_mask.to(data.x.device)

    # For attack evaluation: designate 10% target and 10% clean test nodes
    num_target = int(0.1 * num_nodes)
    target_mask = torch.zeros(num_nodes, dtype=torch.bool)
    clean_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    target_mask[indices[num_train + num_val:num_train + num_val + num_target]] = True
    clean_test_mask[indices[num_train + num_val + num_target:]] = True

    data.target_mask = target_mask.to(data.x.device)
    data.clean_test_mask = clean_test_mask.to(data.x.device)

    return data

def select_high_centrality_nodes(data, num_nodes_to_select):
    graph = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    graph.add_edges_from(edge_index.T)
    centrality = nx.degree_centrality(graph)
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    return torch.tensor(sorted_nodes[:num_nodes_to_select], dtype=torch.long).to(data.x.device)

def generate_subgraphs(data, policy="edge_deleted", max_subgraphs=100):
    """
    Generate subgraphs from the original graph according to the specified policy.
    This is useful for models (like ESAN) that process subgraphs.
    """
    from torch_geometric.utils import from_networkx
    graph = to_networkx(data, to_undirected=True)
    subgraphs = []
    if policy == "edge_deleted":
        for edge in graph.edges:
            if len(subgraphs) >= max_subgraphs:
                break
            subgraph = graph.copy()
            subgraph.remove_edge(*edge)
            pyg_subgraph = from_networkx(subgraph)
            pyg_subgraph.n_id = torch.tensor(list(subgraph.nodes), device=data.x.device)
            if pyg_subgraph.x is None:
                pyg_subgraph.x = data.x[pyg_subgraph.n_id]
            subgraphs.append(pyg_subgraph)
    elif policy == "node_deleted":
        for node in list(graph.nodes):
            if len(subgraphs) >= max_subgraphs:
                break
            subgraph = graph.copy()
            subgraph.remove_node(node)
            pyg_subgraph = from_networkx(subgraph)
            pyg_subgraph.n_id = torch.tensor(list(subgraph.nodes), device=data.x.device)
            if pyg_subgraph.x is None:
                pyg_subgraph.x = data.x[pyg_subgraph.n_id]
            subgraphs.append(pyg_subgraph)
    elif policy == "ego":
        from torch_geometric.utils import from_networkx
        radius = 2
        for node in graph.nodes:
            if len(subgraphs) >= max_subgraphs:
                break
            subgraph = nx.ego_graph(graph, node, radius=radius)
            pyg_subgraph = from_networkx(subgraph)
            pyg_subgraph.n_id = torch.tensor(list(subgraph.nodes), device=data.x.device)
            central_indicator = torch.zeros(len(subgraph.nodes), 1, device=data.x.device)
            node_list = list(subgraph.nodes)
            central_index = node_list.index(node)
            central_indicator[central_index] = 1
            if pyg_subgraph.x is None:
                pyg_subgraph.x = data.x[pyg_subgraph.n_id]
            pyg_subgraph.x = torch.cat([pyg_subgraph.x, central_indicator], dim=1)
            subgraphs.append(pyg_subgraph)
    return subgraphs

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
        
def select_diverse_nodes(data, num_nodes_to_select, num_clusters=None):
    """
    Select nodes using a clustering-based approach to ensure diversity.
    Uses K-means on node features (or embeddings) and selects nodes closest to the cluster centers.
    """
    if num_clusters is None:
        num_clusters = len(torch.unique(data.y))
    embeddings = data.x.cpu().numpy()  # For simplicity, using raw features
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    selected_nodes = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        center = centers[i]
        distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
        closest = cluster_indices[np.argmin(distances)]
        selected_nodes.append(closest)
    return torch.tensor(selected_nodes, dtype=torch.long).to(data.x.device)
