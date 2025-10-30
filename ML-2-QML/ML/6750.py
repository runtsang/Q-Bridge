import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FastBaseEstimator:
    """Evaluate a PyTorch model for batches of parameters and observables."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self, observables, parameter_sets):
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
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
    """Add Gaussian shot‑noise to deterministic estimates."""
    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

class EnhancedSamplerQNN(nn.Module):
    """
    Deep classical sampler network with a unified estimator interface.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        estimator = FastEstimator(self) if shots is not None else FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["EnhancedSamplerQNN"]
