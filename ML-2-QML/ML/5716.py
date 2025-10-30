"""ML component of QCNNPlus – a hybrid convolution‑inspired network with a classical feature extractor and a quantum kernel head.

The seed QCNNModel was a stack of linear layers.  QCNNPlus expands this by:
*  a configurable **feature‑map** module (any nn.Module) that can be swapped for a pretrained ResNet, VGG or a simple MLP;
*  a **quantum kernel** implemented as a Qiskit EstimatorQNN that produces a scalar kernel value for each input pair;
*  a **trainable classifier** that takes the concatenated classical + quantum embeddings and outputs a probability.

The class is fully trainable with PyTorch optimizers and can be used as a downstream classifier in a continuous‑learning pipeline.

The module can be dropped into a normal PyTorch training script – the only extra dependency is ``qiskit_machine_learning``.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit_machine_learning.neural_networks import EstimatorQNN


class FeatureMap(nn.Module):
    """Simple MLP that can be replaced by any pretrained network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    hidden : int, optional
        Size of the hidden layer. Default 32.
    output_dim : int, optional
        Size of the output embedding. If None, defaults to `hidden`.
    """
    def __init__(self, input_dim: int, hidden: int = 32, output_dim: int | None = None):
        super().__init__()
        if output_dim is None:
            output_dim = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumKernelWrapper(nn.Module):
    """Wraps a Qiskit EstimatorQNN as a PyTorch module.

    The wrapper is non‑differentiable; it treats the quantum part as a fixed
    feature extractor.  This is useful for transfer‑learning experiments
    where the classical head is trained while the quantum ansatz is frozen.
    """
    def __init__(self, qnn: EstimatorQNN):
        super().__init__()
        self.qnn = qnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy array on CPU
        inputs = x.detach().cpu().numpy()
        # EstimatorQNN expects shape (batch, input_dim)
        outputs = self.qnn.evaluate(inputs)
        # outputs shape (batch, 1)
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)


class QCNNPlus(nn.Module):
    """Hybrid QCNN model combining a classical feature extractor and a quantum kernel head.

    The architecture is:

        inputs ──> FeatureMap ──> [concatenate] ──> Classifier
                   ▲                     ▲
                QuantumKernel            │
                   │                     │
                (EstimatorQNN)           │
    """
    def __init__(
        self,
        input_dim: int = 8,
        feature_hidden: int = 32,
        feature_output: int = 64,
        qnn: EstimatorQNN | None = None,
        num_classes: int = 1,
    ):
        super().__init__()
        self.feature_extractor = FeatureMap(input_dim, hidden=feature_hidden, output_dim=feature_output)
        if qnn is None:
            raise ValueError("A pre‑built EstimatorQNN instance must be supplied.")
        self.quantum_layer = QuantumKernelWrapper(qnn)
        self.classifier = nn.Sequential(
            nn.Linear(feature_output + 1, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        qfeat = self.quantum_layer(x)
        combined = torch.cat([feat, qfeat], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)


class EarlyStopping:
    """Simple early‑stopping callback for training loops.

    Parameters
    ----------
    patience : int, default 5
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, default 1e-4
        Minimum change in the monitored loss to qualify as an improvement.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


__all__ = [
    "FeatureMap",
    "QuantumKernelWrapper",
    "QCNNPlus",
    "EarlyStopping",
]
