import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score

from dataset import load_dataset, split_dataset, select_high_centrality_nodes, generate_subgraphs
from attacks import inject_trigger, train_with_poisoned_data 
from models import ESAN, SUN, SubstructureAwareGNN, GNN, TriggerGenerator, OODDetector, SubstructureAwareGNN_CS, SproutGNN
from models.ood_detector import train_ood_detector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Determine which model types to run.
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg == "all":
        model_types = ["esan", "sun", "sagn", "sagn+cs", "gnn", "sproutgnn"]
    else:
        model_types = [arg]
else:
    model_types = ["esan", "sun", "sagn", "sagn+cs", "gnn", "sproutgnn"]


def instantiate_model(model_type, input_dim, output_dim):
    if model_type == "esan":
        return ESAN(input_dim, hidden_dim=64, output_dim=output_dim).to(device)
    elif model_type == "sun":
        return SUN(num_features=input_dim, num_classes=output_dim, hidden_channels=64).to(device)
    elif model_type == "sagnn":
        return SubstructureAwareGNN(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
    elif model_type == "sagnn+cs":
        from models import SubstructureAwareGNN_CS
        return SubstructureAwareGNN_CS(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
    elif model_type == "gnn":
        return GNN(input_dim, hidden_dim=64, output_dim=output_dim, model_type='GCN').to(device)
    elif model_type == "sproutgnn":
        return SproutGNN(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# --- Metrics Functions ---
def compute_metrics_esan(model, subgraphs, data, poisoned_nodes, target_labels):
    model.eval()
    with torch.no_grad():
        out = model(subgraphs, data.num_nodes)
        _, pred = out.max(dim=1)
        asr = (pred[poisoned_nodes] == target_labels).sum().item() / len(poisoned_nodes) * 100
        clean_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu()) * 100
    return asr, clean_acc



def compute_metrics(model, data, poisoned_nodes, target_labels):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).detach()
        _, pred = out.max(dim=1)
        asr = (pred[poisoned_nodes] == target_labels).sum().item() / len(poisoned_nodes) * 100
        clean_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu()) * 100
    return asr, clean_acc


# --- Training and Testing for ESAN ---
def train_model_esan(model, subgraphs, data, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(subgraphs, data.num_nodes)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # Print training progress at selected epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            train_acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean().item()
            print(f"ESAN Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}")
    return model


def test_model_esan(model, subgraphs, data):
    model.eval()
    logits = model(subgraphs, data.num_nodes)
    accs = {}
    for mask_name, mask in zip(["Train", "Validation", "Test"],
                               [data.train_mask, data.val_mask, data.test_mask]):
        pred = logits[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean().item()
        accs[mask_name] = acc * 100
        print(f"ESAN {mask_name} Accuracy: {acc * 100:.2f}%")
    return accs


# --- Main Experiment Function ---
def run_attacks_for_model(model_type):
    results_summary = []
    print(f"\n===== Running experiments for model: {model_type.upper()} =====")
    dataset_budgets = {'Cora': 10, 'CiteSeer': 30, 'PubMed': 40}
    
    for dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        dataset = load_dataset(dataset_name)
        data = dataset[0].to(device)
        input_dim = data.num_features
        output_dim = dataset.num_classes if isinstance(dataset.num_classes, int) else dataset.num_classes[0]
        orig_num_nodes = data.num_nodes
        data = split_dataset(data)
        poisoned_node_budget = dataset_budgets.get(dataset_name, 10)
        poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget).to(device)

        if model_type == "esan":
            policies = ["ego", "edge_deleted", "node_deleted"]
            for policy in policies:
                print(f"\nGenerating subgraphs using policy '{policy}' for {dataset_name}")
                subgraphs = generate_subgraphs(data, policy=policy, max_subgraphs=200)
                adjusted_input_dim = input_dim + 1 if policy == "ego" else input_dim
                print(f"Training baseline ESAN model for {dataset_name} with policy '{policy}'")
                model_inst = ESAN(adjusted_input_dim, hidden_dim=64, output_dim=output_dim).to(device)
                optimizer = torch.optim.Adam(model_inst.parameters(), lr=0.01)
                model_inst = train_model_esan(model_inst, subgraphs, data, optimizer, epochs=200)
                baseline_accs = test_model_esan(model_inst, subgraphs, data)
                baseline_test_acc = baseline_accs["Test"]
                print(f"Dataset: {dataset_name}, Policy: {policy}, Baseline Test Accuracy: {baseline_test_acc:.2f}%")
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": "ESAN",
                    "Policy": policy,
                    "Attack": "None",
                    "ASR": "N/A",
                    "Clean Accuracy": baseline_test_acc
                })
                attack_methods = ['SBA-Samp', 'SBA-Gen', 'GTA', 'UGBA', 'DPGBA']
                for attack in attack_methods:
                    try:
                        print(f"\nStarting attack {attack} on {dataset_name} with ESAN (Policy: {policy})")
                        model_inst = ESAN(adjusted_input_dim, hidden_dim=256, output_dim=output_dim).to(device)
                        optimizer = torch.optim.Adam(model_inst.parameters(), lr=0.002)
                        poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget).to(device)
                        if attack == "DPGBA":
                            print("Initializing trigger generator and OOD detector for DPGBA")
                            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)
                            ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                            ood_optimizer = torch.optim.Adam(ood_detector.parameters(), lr=0.001)
                            train_ood_detector(ood_detector, data, ood_optimizer, epochs=100)
                        else:
                            trigger_gen = None
                            ood_detector = None
                        data_poisoned, target_labels = inject_trigger(
                            data=data,
                            poisoned_nodes=poisoned_nodes,
                            attack_type=attack,
                            model=model_inst,
                            trigger_gen=trigger_gen,
                            ood_detector=ood_detector,
                            alpha=0.7,
                            trigger_size=3,
                            trigger_density=0.5
                        )
                        subgraphs_poisoned = generate_subgraphs(data_poisoned, policy=policy, max_subgraphs=100)
                        model_inst = train_model_esan(model_inst, subgraphs_poisoned, data_poisoned, optimizer, epochs=200)
                        asr, clean_acc = compute_metrics_esan(model_inst, subgraphs_poisoned, data_poisoned, poisoned_nodes, target_labels)
                        results_summary.append({
                            "Dataset": dataset_name,
                            "Model": "ESAN",
                            "Policy": policy,
                            "Attack": attack,
                            "ASR": asr,
                            "Clean Accuracy": clean_acc
                        })
                        print(f"Dataset: {dataset_name}, ESAN (Policy: {policy}), Attack: {attack} - ASR: {asr:.2f}%, Clean Accuracy: {clean_acc:.2f}%")
                    except Exception as e:
                        print(f"Error during attack {attack} on {dataset_name} with ESAN (Policy: {policy}): {e}")

        elif model_type == "sproutgnn":
            print(f"Initializing SproutGNN for {dataset_name}")
            model = SproutGNN(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

            # Baseline training
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.y)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(f"{dataset_name} | SproutGNN | Base | Epoch {epoch:03d} | Loss {loss.item():.4f}")

            # Baseline evaluation
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.y)
                pred = out.argmax(dim=1)
                baseline_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item() * 100
            print(f"{dataset_name} | SproutGNN | Base | Acc {baseline_acc:.2f}%")
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "SproutGNN",
                "Attack": "None",
                "Defense": "None",
                "ASR": "N/A",
                "Clean Accuracy": baseline_acc
            })

            # Poisoning & attack loop
            print(f"Initializing Trigger Generator for {dataset_name}")
            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)

            for attack in attack_methods:
                try:
                    print(f"Starting attack {attack} on {dataset_name} with SproutGNN")
                    # Reinitialize model & optimizer per attack
                    model = SproutGNN(in_channels=input_dim, hidden_channels=256, out_channels=output_dim).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

                    # DPGBA setup
                    if attack == "DPGBA":
                        print("Initializing OOD Detector for DPGBA")
                        ood = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                        ood_optimizer = torch.optim.Adam(ood.parameters(), lr=0.001)
                        train_ood_detector(ood, data, ood_optimizer, epochs=100)
                        current_trigger_gen, ood_detector = trigger_gen, ood
                    else:
                        current_trigger_gen, ood_detector = None, None

                    data_poisoned, target_labels = inject_trigger(
                        data=data,
                        poisoned_nodes=poisoned_nodes,
                        attack_type=attack,
                        model=model,
                        trigger_gen=current_trigger_gen,
                        ood_detector=ood_detector,
                        alpha=0.7,
                        trigger_size=3,
                        trigger_density=0.5
                    )

                    # Train on poisoned graph
                    model.train()
                    for epoch in range(200):
                        optimizer.zero_grad()
                        out = model(data_poisoned.x, data_poisoned.edge_index, data_poisoned.y)
                        loss = F.nll_loss(
                            out[data_poisoned.train_mask],
                            data_poisoned.y[data_poisoned.train_mask]
                        )
                        loss.backward()
                        optimizer.step()
                        if epoch % 10 == 0:
                            print(f"{dataset_name} | SproutGNN | {attack} | Epoch {epoch:03d} | Loss {loss.item():.4f}")

                    # Evaluate
                    asr, clean_acc = compute_metrics(
                        model, data_poisoned,
                        poisoned_nodes, target_labels,
                        use_y=True
                    )
                    print(f"{dataset_name} | SproutGNN | {attack} | ASR {asr:.2f}%, Acc {clean_acc:.2f}%")
                    results_summary.append({
                        "Dataset": dataset_name,
                        "Model": "SproutGNN",
                        "Attack": attack,
                        "Defense": "None",
                        "ASR": asr,
                        "Clean Accuracy": clean_acc
                    })
                except Exception as e:
                    print(f"Error during attack {attack} on {dataset_name} with SproutGNN: {e}")

        elif model_type == "sagnn":
            print(f"Initializing SAGNN for dataset {dataset_name}")
            model = SubstructureAwareGNN(in_channels=input_dim, hidden_channels=256, out_channels=output_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(f"Dataset: {dataset_name}, SAGN Epoch {epoch}, Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                predictions = out.argmax(dim=1)
                baseline_acc = ((predictions[data.test_mask] == data.y[data.test_mask]).sum().item() /
                                data.test_mask.sum().item())
            print(f"Dataset: {dataset_name}, Model: SAGN, Baseline Accuracy: {baseline_acc * 100:.2f}%")
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "SAGNN",
                "Attack": "None",
                "Defense": "None",
                "ASR": "N/A",
                "Clean Accuracy": baseline_acc * 100
            })
            print(f"Initializing Trigger Generator for dataset: {dataset_name}")
            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)
            print(f"Selecting poisoned nodes for dataset: {dataset_name}")
            poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget).to(device)
            print(f"Selected {len(poisoned_nodes)} poisoned nodes.")
            attack_methods = ['SBA-Samp', 'SBA-Gen', 'GTA', 'UGBA', 'DPGBA']
            for attack in attack_methods:
                try:
                    print(f"Starting attack {attack} on dataset: {dataset_name}")
                    model = SubstructureAwareGNN(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                    if attack == 'DPGBA':
                        print("Initializing OOD Detector for DPGBA")
                        ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                        ood_optimizer = torch.optim.Adam(ood_detector.parameters(), lr=0.001)
                        train_ood_detector(ood_detector, data, ood_optimizer, epochs=100)
                        current_trigger_gen = trigger_gen
                    else:
                        ood_detector = None
                        current_trigger_gen = None
                    data_poisoned, target_labels = inject_trigger(
                        data=data,
                        poisoned_nodes=poisoned_nodes,
                        attack_type=attack,
                        model=model,
                        trigger_gen=current_trigger_gen,
                        ood_detector=ood_detector,
                        alpha=0.7,
                        trigger_size=3,
                        trigger_density=0.5
                    )
                    model.train()
                    for epoch in range(200):
                        optimizer.zero_grad()
                        out = model(data_poisoned.x, data_poisoned.edge_index)
                        loss = F.cross_entropy(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
                        loss.backward()
                        optimizer.step()
                        if epoch % 10 == 0:
                            print(f"Dataset: {dataset_name}, SAGN Attack {attack}, Epoch {epoch}, Loss: {loss.item():.4f}")
                    asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes, target_labels)
                    results_summary.append({
                        "Dataset": dataset_name,
                        "Model": "SAGNN",
                        "Attack": attack,
                        "Defense": "None",
                        "ASR": asr,
                        "Clean Accuracy": clean_acc
                    })
                    print(f"Dataset: {dataset_name}, Attack: {attack} - ASR: {asr:.2f}%, Clean Accuracy: {clean_acc:.2f}%")
                except Exception as e:
                    print(f"Error during attack {attack} on dataset {dataset_name} with SAGN: {e}")

        elif model_type == "sagnn+cs":
            print(f"Initializing SAGNN+CS for dataset {dataset_name}")
            model = SubstructureAwareGNN_CS(in_channels=input_dim, hidden_channels=256, out_channels=output_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(f"Dataset: {dataset_name}, SAGNN+CS Epoch {epoch}, Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                predictions = out.argmax(dim=1)
                baseline_acc = ((predictions[data.test_mask] == data.y[data.test_mask]).sum().item() /
                                data.test_mask.sum().item())
            print(f"Dataset: {dataset_name}, Model: SAGNN+CS, Baseline Accuracy: {baseline_acc * 100:.2f}%")
            results_summary.append({
                "Dataset": dataset_name,
                "Model": "SAGNN+CS",
                "Attack": "None",
                "Defense": "None",
                "ASR": "N/A",
                "Clean Accuracy": baseline_acc * 100
            })
            print(f"Initializing Trigger Generator for dataset: {dataset_name}")
            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)
            print(f"Selecting poisoned nodes for dataset: {dataset_name}")
            poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget).to(device)
            print(f"Selected {len(poisoned_nodes)} poisoned nodes.")
            attack_methods = ['SBA-Samp', 'SBA-Gen', 'GTA', 'UGBA', 'DPGBA']
            for attack in attack_methods:
                try:
                    print(f"Starting attack {attack} on dataset: {dataset_name}")
                    model = SubstructureAwareGNN_CS(in_channels=input_dim, hidden_channels=64, out_channels=output_dim).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                    if attack == 'DPGBA':
                        print("Initializing OOD Detector for DPGBA")
                        ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                        ood_optimizer = torch.optim.Adam(ood_detector.parameters(), lr=0.001)
                        train_ood_detector(ood_detector, data, ood_optimizer, epochs=100)
                        current_trigger_gen = trigger_gen
                    else:
                        ood_detector = None
                        current_trigger_gen = None
                    data_poisoned, target_labels = inject_trigger(
                        data=data,
                        poisoned_nodes=poisoned_nodes,
                        attack_type=attack,
                        model=model,
                        trigger_gen=current_trigger_gen,
                        ood_detector=ood_detector,
                        alpha=0.7,
                        trigger_size=3,
                        trigger_density=0.5
                    )
                    model.train()
                    for epoch in range(200):
                        optimizer.zero_grad()
                        out = model(data_poisoned.x, data_poisoned.edge_index)
                        loss = F.cross_entropy(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
                        loss.backward()
                        optimizer.step()
                        if epoch % 10 == 0:
                            print(f"Dataset: {dataset_name}, SAGN+CS Attack {attack}, Epoch {epoch}, Loss: {loss.item():.4f}")
                    asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes, target_labels)
                    results_summary.append({
                        "Dataset": dataset_name,
                        "Model": "SAGNN+CS",
                        "Attack": attack,
                        "Defense": "None",
                        "ASR": asr,
                        "Clean Accuracy": clean_acc
                    })
                    print(f"Dataset: {dataset_name}, Attack: {attack} - ASR: {asr:.2f}%, Clean Accuracy: {clean_acc:.2f}%")
                except Exception as e:
                    print(f"Error during attack {attack} on dataset {dataset_name} with SAGN+CS: {e}")

        elif model_type == "sun":
            print(f"Initializing SUN for dataset {dataset_name}")
            model = SUN(num_features=input_dim, num_classes=output_dim, hidden_channels=256).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            print(f"Training baseline model for dataset: {dataset_name}")
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    print(f"Dataset: {dataset_name}, SUN Epoch {epoch}, Loss: {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                predictions = out.argmax(dim=1)
                baseline_acc = ((predictions[data.test_mask] == data.y[data.test_mask]).sum().item() /
                                data.test_mask.sum().item())
                print(f"Dataset: {dataset_name}, Baseline Accuracy: {baseline_acc * 100:.2f}%")
                results_summary.append({
                    "Dataset": dataset_name,
                    "Model": "SUN",
                    "Attack": "None",
                    "Defense": "None",
                    "ASR": "N/A",
                    "Clean Accuracy": baseline_acc * 100
                })
            print(f"Initializing Trigger Generator for dataset: {dataset_name}")
            trigger_gen = TriggerGenerator(input_dim=input_dim, hidden_dim=64).to(device)
            print(f"Selecting poisoned nodes for dataset: {dataset_name}")
            poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget).to(device)
            print(f"Selected {len(poisoned_nodes)} poisoned nodes.")
            attack_methods = ['SBA-Samp', 'SBA-Gen', 'GTA', 'UGBA', 'DPGBA']
            for attack in attack_methods:
                try:
                    print(f"Starting attack {attack} on dataset: {dataset_name}")
                    if attack == 'DPGBA':
                        print("Initializing OOD Detector for DPGBA")
                        ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                        ood_optimizer = torch.optim.Adam(ood_detector.parameters(), lr=0.001)
                        train_ood_detector(ood_detector, data, ood_optimizer, epochs=100)
                        current_trigger_gen = trigger_gen
                    else:
                        ood_detector = None
                        current_trigger_gen = None
                    data_poisoned, target_labels = inject_trigger(
                        data=data,
                        poisoned_nodes=poisoned_nodes,
                        attack_type=attack,
                        model=model,
                        trigger_gen=current_trigger_gen,
                        ood_detector=ood_detector,
                        alpha=0.7,
                        trigger_size=3,
                        trigger_density=0.5
                    )
                    model.train()
                    for epoch in range(200):
                        optimizer.zero_grad()
                        out = model(data_poisoned.x, data_poisoned.edge_index)
                        loss = F.cross_entropy(out[data_poisoned.train_mask], data_poisoned.y[data_poisoned.train_mask])
                        loss.backward()
                        optimizer.step()
                        if epoch % 10 == 0:
                            print(f"Dataset: {dataset_name}, SUN Attack {attack}, Epoch {epoch}, Loss: {loss.item():.4f}")
                    asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes, target_labels)
                    results_summary.append({
                        "Dataset": dataset_name,
                        "Model": "SUN",
                        "Attack": attack,
                        "Defense": "None",
                        "ASR": asr,
                        "Clean Accuracy": clean_acc
                    })
                    print(f"Dataset: {dataset_name}, Attack: {attack} - ASR: {asr:.2f}%, Clean Accuracy: {clean_acc:.2f}%")
                except Exception as e:
                    print(f"Error during attack {attack} on dataset {dataset_name}: {e}")

        elif model_type == "gnn":
            gnn_types = ['GCN', 'GraphSage', 'GAT']
            for gnn_type in gnn_types:
                try:
                    print(f"Initializing {gnn_type} for dataset {dataset_name}")
                    model = GNN(input_dim=input_dim, hidden_dim=64, output_dim=output_dim, model_type=gnn_type).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
                    model.train()
                    for epoch in range(200):
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index)
                        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        optimizer.step()
                        if epoch % 10 == 0:
                            print(f"Dataset: {dataset_name}, {gnn_type} Epoch {epoch}, Loss: {loss.item():.4f}")
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        predictions = out.argmax(dim=1)
                        baseline_acc = ((predictions[data.test_mask] == data.y[data.test_mask]).sum().item() /
                                        data.test_mask.sum().item())
                        print(f"Dataset: {dataset_name}, Model: {gnn_type}, Baseline Accuracy: {baseline_acc * 100:.2f}%")
                        results_summary.append({
                            "Dataset": dataset_name,
                            "Model": gnn_type,
                            "Attack": "None",
                            "ASR": "N/A",
                            "Clean Accuracy": baseline_acc * 100
                        })
                    trigger_gen = TriggerGenerator(input_dim=data.num_features, hidden_dim=64).to(device)
                    ood_detector = OODDetector(input_dim=input_dim, hidden_dim=64, latent_dim=16).to(device)
                    ood_optimizer = torch.optim.Adam(ood_detector.parameters(), lr=0.001)
                    train_ood_detector(ood_detector, data, ood_optimizer, epochs=100)
                    poisoned_nodes = select_high_centrality_nodes(data, poisoned_node_budget)
                    attack_methods = ['SBA-Samp', 'SBA-Gen', 'GTA', 'UGBA', 'DPGBA']
                    for attack in attack_methods:
                        try:
                            print(f"Running attack {attack} on {gnn_type} for dataset {dataset_name}")
                            trained_model, data_poisoned = train_with_poisoned_data(
                                model=model,
                                data=data,
                                optimizer=optimizer,
                                poisoned_nodes=poisoned_nodes,
                                trigger_gen=trigger_gen,
                                attack=attack,
                                ood_detector=ood_detector,
                                alpha=0.7,
                                early_stopping=True
                            )
                            asr, clean_acc = compute_metrics(model, data_poisoned, poisoned_nodes, target_labels)
                            results_summary.append({
                                "Dataset": dataset_name,
                                "Model": gnn_type,
                                "Attack": attack,
                                "ASR": asr,
                                "Clean Accuracy": clean_acc
                            })
                            print(f"Dataset: {dataset_name}, Model: {gnn_type}, Attack: {attack} - ASR: {asr:.2f}%, Clean Accuracy: {clean_acc:.2f}%")
                        except Exception as e:
                            print(f"Error during attack {attack} on dataset {dataset_name} with model {gnn_type}: {e}")
                            continue
                except Exception as e:
                    print(f"Error with model {gnn_type} on dataset {dataset_name}: {e}")
                    continue
        else:
            print(f"Unknown model type: {model_type}")
            continue

    return results_summary


def run_attacks_on_model():
    """
    Orchestrate attack runs for each model type, gather results, and save to separate CSV files.
    """
    for mtype in model_types:
        results = run_attacks_for_model(mtype)
        results_df = pd.DataFrame(results)
        filename = f"backdoor_attack_results_summary_{mtype}.csv"
        print(f"\nSummary of results for {mtype}:")
        print(results_df)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


if __name__ == "__main__":
    run_attacks_on_model()
