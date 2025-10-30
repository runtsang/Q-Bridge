import torch
from torch import nn
import numpy as np

class HybridNATModel(nn.Module):
    """Classical CNN backbone with optional quantum augmentation."""
    def __init__(self, n_classes=4, use_quantum=False, quantum_layer=None):
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_layer = quantum_layer
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.use_quantum and self.quantum_layer is not None:
            q_out = self.quantum_layer(x)
            # Assume quantum_layer returns a tensor of shape (batch, n_quantum)
            x = torch.cat([x, q_out], dim=1)
        x = self.fc(x)
        return self.norm(x)

    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        """Batch evaluation with optional shot noise."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy
