import torch
from torch import nn
import numpy as np
from.FastBaseEstimator import FastEstimator

class HybridQuantumLayer(nn.Module):
    """Hybrid classical‑quantum inspired layer combining convolution, quantum‑style linear block and QCNN stack."""
    def __init__(self):
        super().__init__()
        # Classical convolution to mimic QuantumNAT feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum‑style linear block: a few dense layers with non‑linearities
        self.quantum_block = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # QCNN‑style fully connected stack
        self.qcnn_stack = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 24), nn.Tanh(),
            nn.Linear(24, 16), nn.Tanh(),
            nn.Linear(16, 8), nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.quantum_block(x)
        x = self.qcnn_stack(x)
        return self.norm(x)

    def evaluate(self, observables, parameter_sets, shots=None, seed=None):
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

def FCL():
    """Factory returning the hybrid layer."""
    return HybridQuantumLayer()

__all__ = ["HybridQuantumLayer", "FCL"]
