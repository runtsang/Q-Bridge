import torch
import torch.nn as nn
import numpy as np
from quantum_head import QuantumHybridLayer

class UniversalEstimatorQNN(nn.Module):
    """
    Hybrid estimator that merges a classical backbone (FC or CNN) with a
    variational quantum expectation layer. Supports regression and
    binary classification.
    """
    def __init__(self,
                 task: str = "regression",
                 backbone: str = "fc",
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (8, 4),
                 n_cnn_layers: int = 3,
                 n_channels: int = 3,
                 img_size: tuple[int, int] = (32, 32),
                 n_qubits: int = 2,
                 shift: float = np.pi/2,
                 shots: int = 100,
                 device: str = "cpu"):
        super().__init__()
        self.task = task
        self.backbone_type = backbone
        self.n_qubits = n_qubits
        self.shift = shift
        self.shots = shots
        self.device = device

        # Classical backbone
        if backbone == "fc":
            layers = []
            in_features = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_features, h))
                layers.append(nn.Tanh())
                in_features = h
            layers.append(nn.Linear(in_features, n_qubits))
            self.backbone = nn.Sequential(*layers)
        else:  # cnn
            self.backbone = nn.Sequential(
                nn.Conv2d(n_channels, 6, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Flatten(),
                nn.Linear(55815, 120),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(84, n_qubits),
            )

        # Quantum expectation head
        self.quantum_head = QuantumHybridLayer(n_qubits, shift, shots, device)

        # Final head
        self.final_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        q_out = self.quantum_head(features)
        out = self.final_head(q_out)
        if self.task == "classification":
            probs = torch.sigmoid(out)
            return torch.cat([probs, 1 - probs], dim=-1)
        else:
            return out.squeeze(-1)
