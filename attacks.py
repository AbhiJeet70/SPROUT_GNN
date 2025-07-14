# attacks.py
import torch
import random
import torch.nn.functional as F
import networkx as nx

from dataset import select_diverse_nodes 

def inject_trigger(data, poisoned_nodes, attack_type, model, trigger_gen=None, ood_detector=None,
                   alpha=0.7, trigger_size=5, trigger_density=0.5, input_dim=None):
    # Clone data to avoid overwriting the original graph
    data_poisoned = data.clone()
    device = data_poisoned.x.device

    if len(poisoned_nodes) == 0:
        raise ValueError("No poisoned nodes selected. Ensure 'poisoned_nodes' is populated and non-empty.")

    # Adjust trigger_size if it exceeds the number of poisoned nodes
    trigger_size = min(trigger_size, len(poisoned_nodes))

    # Initialize target labels as a copy of the original labels for the poisoned nodes.
    target_labels = data.y[poisoned_nodes].clone()

    if attack_type == 'SBA-Samp':
        # Subgraph-Based Attack - Random Sampling
        connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes[:trigger_size]]
        avg_features = torch.stack([
            data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes
        ])
        natural_features = avg_features + torch.randn_like(avg_features) * 0.02  # Small randomness

        # Generate subgraph with realistic density
        import networkx as nx
        G = nx.erdos_renyi_graph(trigger_size, trigger_density)
        trigger_edge_index = torch.tensor(list(G.edges)).t().contiguous()

        # Connect poisoned nodes to the subgraph
        poisoned_edges = torch.stack([
            poisoned_nodes[:trigger_size],
            torch.randint(0, data.num_nodes, (trigger_size,), device=device)
        ])

        # Update graph structure and features
        data_poisoned.edge_index = torch.cat([data.edge_index, trigger_edge_index.to(device),
                                                poisoned_edges.to(device)], dim=1)
        data_poisoned.x[poisoned_nodes[:trigger_size]] = natural_features[:trigger_size]

        # Modify target labels - Random misclassification
        num_classes = data.y.max().item() + 1
        for i in range(len(poisoned_nodes)):
            original_label = data.y[poisoned_nodes[i]].item()
            possible_labels = list(set(range(num_classes)) - {original_label})
            target_labels[i] = random.choice(possible_labels)

    elif attack_type == 'SBA-Gen':
        # Subgraph-Based Attack - Gaussian
        connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes[:trigger_size]]

        # Calculate mean and standard deviation of features
        feature_mean = data.x.mean(dim=0)
        feature_std = data.x.std(dim=0)

        # Generate natural features with Gaussian noise
        avg_features = torch.stack([
            data.x[nodes].mean(dim=0) if len(nodes) > 0 else feature_mean for nodes in connected_nodes
        ])
        natural_features = avg_features + torch.normal(mean=0.0, std=0.03, size=avg_features.shape).to(data.x.device)

        # Generate subgraph edges based on Gaussian similarity
        trigger_edge_index = []
        for i in range(trigger_size):
            for j in range(i + 1, trigger_size):
                similarity = torch.exp(-torch.norm((natural_features[i] - natural_features[j]) / feature_std) ** 2)
                if similarity > torch.rand(1).item():
                    trigger_edge_index.append([i, j])
        if trigger_edge_index:
            trigger_edge_index = torch.tensor(trigger_edge_index, dtype=torch.long).t().contiguous()
            trigger_edge_index += poisoned_nodes[:trigger_size].unsqueeze(0)
        else:
            trigger_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        poisoned_edges = torch.stack([
            poisoned_nodes[:trigger_size],
            torch.randint(0, data.num_nodes, (trigger_size,), device=device)
        ])

        data_poisoned.edge_index = torch.cat([data.edge_index, trigger_edge_index.to(device),
                                                poisoned_edges.to(device)], dim=1)
        data_poisoned.x[poisoned_nodes[:trigger_size]] = natural_features[:trigger_size]

        num_classes = data.y.max().item() + 1
        for i in range(len(poisoned_nodes)):
            original_label = data.y[poisoned_nodes[i]].item()
            possible_labels = list(set(range(num_classes)) - {original_label})
            target_labels[i] = random.choice(possible_labels)

    elif attack_type == 'DPGBA':
        # DPGBA Attack: Distribution-Preserving Graph Backdoor Attack
        connected_nodes = [data.edge_index[0][data.edge_index[1] == node] for node in poisoned_nodes]
        avg_features = torch.stack([
            data.x[nodes].mean(dim=0) if len(nodes) > 0 else data.x.mean(dim=0) for nodes in connected_nodes
        ]).to(device)

        if trigger_gen is None:
            raise ValueError("Trigger generator is required for the DPGBA attack.")

        with torch.no_grad():
            trigger_features = trigger_gen(avg_features)

        if trigger_features.shape[1] != data.x.shape[1]:
            raise ValueError(f"Trigger feature dimension mismatch: {trigger_features.shape[1]} vs {data.x.shape[1]}")

        node_alphas = torch.rand(len(poisoned_nodes)).to(device) * 0.3 + 0.5
        distribution_preserved_features = (
            node_alphas.unsqueeze(1) * data.x[poisoned_nodes] +
            (1 - node_alphas.unsqueeze(1)) * trigger_features
        )

        num_classes = data.y.max().item() + 1
        target_labels = (data.y[poisoned_nodes] + 1) % num_classes
        data_poisoned.x[poisoned_nodes] = distribution_preserved_features

    elif attack_type == 'GTA':
        # Graph Trojan Attack with Bi-Level Optimization
        # Determine the target label as the least frequent label among poisoned nodes.
        original_poisoned_labels = data.y[poisoned_nodes]
        unique, counts = torch.unique(original_poisoned_labels, return_counts=True)
        target_class = unique[torch.argmin(counts)].item()  # Choose the least frequent label
        num_poisoned = len(poisoned_nodes)
        # Initialize learnable trigger parameters (per poisoned node)
        trigger_params = torch.zeros((num_poisoned, data.x.size(1)), device=device, requires_grad=True)
        trigger_lr = 0.01
        optimizer_trigger = torch.optim.Adam([trigger_params], lr=trigger_lr)
        inner_steps = 100  # Increased number of optimization steps
        alpha_fixed = 0.9  # Increased blending factor to 0.9

        # Use the passed model for bi-level optimization
        model.eval()  # Ensure the model is in evaluation mode.
        for step in range(inner_steps):
            optimizer_trigger.zero_grad()
            blended_features = (1 - alpha_fixed) * data.x[poisoned_nodes] + alpha_fixed * trigger_params
            x_modified = data.x.clone()
            x_modified[poisoned_nodes] = blended_features
            out_modified = model(x_modified, data.edge_index)
            target_labels_fixed = torch.full((num_poisoned,), target_class, dtype=torch.long, device=device)
            loss_backdoor = F.cross_entropy(out_modified[poisoned_nodes], target_labels_fixed)
            loss_reg = 0.0003 * torch.norm(trigger_params, p=2)  # Reduced regularization weight
            loss_total = loss_backdoor + loss_reg
            loss_total.backward()
            optimizer_trigger.step()
            if step % 10 == 0:
                print(f"GTA Inner Loop Step {step}: Loss = {loss_total.item():.4f}")
        optimized_trigger = trigger_params.detach()
        data_poisoned = data.clone()
        data_poisoned.x[poisoned_nodes] = (1 - alpha_fixed) * data.x[poisoned_nodes] + alpha_fixed * optimized_trigger
        target_labels = torch.full((num_poisoned,), target_class, dtype=torch.long, device=device)

    elif attack_type == 'UGBA':
        # Unnoticeable Graph Backdoor Attack
        diverse_nodes = select_diverse_nodes(data_poisoned, len(poisoned_nodes)).to(device)
        connected_nodes = [data_poisoned.edge_index[0][data_poisoned.edge_index[1] == node] for node in diverse_nodes]
        avg_features = torch.stack([
            data_poisoned.x[nodes].mean(dim=0) if len(nodes) > 0 else data_poisoned.x.mean(dim=0) for nodes in connected_nodes
        ])
        # Reduced noise magnitude: mean changed to 0.1 and std to 0.05
        refined_trigger_features = avg_features + torch.normal(mean=0.1, std=0.05, size=avg_features.shape).to(data_poisoned.x.device)
        data_poisoned.x[diverse_nodes] = refined_trigger_features.to(data_poisoned.x.device)
        new_edges = []
        for i in range(len(diverse_nodes)):
            node = diverse_nodes[i]
            neighbor = connected_nodes[i][0] if len(connected_nodes[i]) > 0 else diverse_nodes[(i + 1) % len(diverse_nodes)]
            new_edges.append([node, neighbor])
        new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(data_poisoned.x.device)
        data_poisoned.edge_index = torch.cat([data_poisoned.edge_index, new_edges], dim=1)
        # Determine target label as the least frequent among the original poisoned nodes.
        original_poisoned_labels = data.y[poisoned_nodes]
        unique, counts = torch.unique(original_poisoned_labels, return_counts=True)
        target_class = unique[torch.argmin(counts)].item()
        target_labels = torch.full((len(poisoned_nodes),), target_class, dtype=torch.long, device=device)

    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    return data_poisoned, target_labels


def train_with_poisoned_data(model, data, optimizer, poisoned_nodes, trigger_gen, attack, ood_detector=None, alpha=0.7, early_stopping=False):
    data_poisoned, _ = inject_trigger(data, poisoned_nodes, attack, model, trigger_gen, ood_detector, alpha)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data_poisoned.x, data_poisoned.edge_index)
        loss = F.cross_entropy(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
        loss.backward()
        optimizer.step()
        if early_stopping and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model, data_poisoned
