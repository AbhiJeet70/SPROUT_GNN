# models/ood_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.autoencoder import GAE

class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

class OODDetector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(OODDetector, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)
        self.gae = GAE(self.encoder, self.decoder)

    def forward(self, x, edge_index):
        z = self.gae.encode(x, edge_index)
        return z

    def reconstruct(self, z, edge_index):
        return self.decoder(z, edge_index)

    def reconstruction_loss(self, x, edge_index):
        z = self.gae.encode(x, edge_index)
        reconstructed = self.reconstruct(z, edge_index)
        return F.mse_loss(reconstructed, x)

    def detect_ood(self, x, edge_index, threshold):
        loss = self.reconstruction_loss(x, edge_index)
        return loss > threshold

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        z = F.relu(self.conv1(x, edge_index))
        z = self.conv2(z, edge_index)
        return z

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(latent_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, input_dim)

    def forward(self, z, edge_index):
        x = F.relu(self.conv1(z, edge_index))
        x = self.conv2(x, edge_index)
        return x



def train_ood_detector(ood_detector, data, optimizer, epochs=50):
    ood_detector.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass
        z = ood_detector(data.x, data.edge_index)

        # Reconstruct data using the latent embedding
        reconstructed_x = ood_detector.reconstruct(z, data.edge_index)

        # Use only the training mask to compute reconstruction loss
        loss = F.mse_loss(reconstructed_x[data.train_mask], data.x[data.train_mask])
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Reconstruction Loss: {loss.item():.4f}")
