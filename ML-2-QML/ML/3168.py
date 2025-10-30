import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum helper.
    Returns an nn.Sequential network, the encoding indices, weight sizes and observable indices.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridClassifierHead(nn.Module):
    """
    Wrapper around the classifier created by ``build_classifier_circuit``.
    Produces a 2‑class probability distribution.
    """
    def __init__(self, num_features: int, depth: int = 4):
        super().__init__()
        self.classifier, self.encoding, self.weight_sizes, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)


class HybridQuantumBinaryClassifier(nn.Module):
    """
    Classical CNN followed by a lightweight feed‑forward head that emulates the quantum
    expectation layer.  The architecture is fully differentiable with PyTorch.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Flattened feature size for 32x32 RGB images (adjust if needed)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 features feed into the quantum head

        # Head that emulates the quantum expectation layer
        self.classifier_head = HybridClassifierHead(num_features=self.fc3.out_features)

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

        probs = self.classifier_head(x)
        return probs


__all__ = ["HybridQuantumBinaryClassifier", "build_classifier_circuit", "HybridClassifierHead"]
