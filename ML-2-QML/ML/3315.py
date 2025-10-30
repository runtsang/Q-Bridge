import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classifier_circuit(num_features: int, depth: int):
    """Create a purely classical feed‑forward classifier matching the quantum metadata."""
    layers = []
    in_dim = num_features
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    return network, weight_sizes

class HybridQuantumClassifier(nn.Module):
    """Purely classical implementation of the hybrid architecture."""
    def __init__(self, num_features=3, depth=2, device='cpu'):
        super().__init__()
        self.device = device
        # Convolutional backbone
        self.conv1 = nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Classical classifier head
        self.classifier, _ = build_classifier_circuit(1, depth)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridQuantumClassifier"]
