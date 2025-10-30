import torch
import torch.nn as nn
import numpy as np
from.QuantumNAT__gen210_qml import QuantumLayer

class HybridQuantumNAT(nn.Module):
    """Hybrid CNN–quantum model that fuses classical feature extraction and variational quantum layers."""
    def __init__(self, input_dim: int = 8, quantum_wires: int = 4, num_classes: int = 1):
        super().__init__()
        # Classical encoder: linear blocks mimicking the QCNN linear stacks
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh()
        )
        # Quantum variational layer
        self.quantum = QuantumLayer(n_wires=quantum_wires)
        # Final classification/regression head
        self.head = nn.Linear(quantum_wires, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).
        Returns
        -------
        torch.Tensor
            Output of shape (batch, num_classes).
        """
        features = self.encoder(x)
        qout = self.quantum(features)
        return self.head(qout)

# ----------------------------------------------------------------------
# Classical evaluation utilities (adapted from FastBaseEstimator)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate a neural network for batches of inputs and observables."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self, observables, parameter_sets):
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""
    def evaluate(self, observables, parameter_sets, *, shots: int | None = None, seed: int | None = None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQuantumNAT", "FastBaseEstimator", "FastEstimator"]
