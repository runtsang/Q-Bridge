"""Quantum hybrid quanvolution with variational circuit and flexible head."""
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumFilter(nn.Module):
    def __init__(self, num_wires=4, num_layers=2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 3))
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, params):
            for i in range(num_wires):
                qml.RY(inputs[:, i], wires=i)
            for layer in range(num_layers):
                for i in range(num_wires):
                    qml.RY(params[layer, i, 0], wires=i)
                for i in range(num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(num_wires):
                    qml.RY(params[layer, i, 1], wires=i)
                for i in range(num_wires - 1):
                    qml.CNOT(wires=[i + 1, i])
                for i in range(num_wires):
                    qml.RZ(params[layer, i, 2], wires=i)
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(num_wires)], dim=1)
        self.circuit = circuit

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        batch = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2].view(batch, -1)
                patch = (patch - patch.mean()) / (patch.std() + 1e-8)
                patch = patch * torch.pi
                feat = self.circuit(patch, self.params)
                patches.append(feat)
        return torch.cat(patches, dim=1)

class QuanvolutionHybrid(nn.Module):
    def __init__(self, num_wires=4, num_layers=2, hidden_dim=256, num_classes=10, task='classification'):
        super().__init__()
        self.qfilter = QuantumFilter(num_wires, num_layers)
        self.feature_dim = (28 // 2) ** 2 * num_wires
        self.fc = nn.Linear(self.feature_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes if task == 'classification' else hidden_dim)
        self.task = task

    def forward(self, x):
        features = self.qfilter(x)
        features = features.view(features.size(0), -1)
        features = F.relu(self.fc(features))
        logits = self.head(features)
        if self.task == 'classification':
            return F.log_softmax(logits, dim=-1)
        else:
            return logits

QuanvolutionFilter = QuanvolutionHybrid
QuanvolutionClassifier = QuanvolutionHybrid
