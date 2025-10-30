import torch
from torch import nn
import numpy as np
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class ClassicalFeatureExtractor(nn.Module):
    """Conv‑FC backbone inspired by Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
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
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class CombinedEstimatorQNN(nn.Module):
    """
    Hybrid estimator that concatenates classical convolutional features with a
    Qiskit variational circuit.  The quantum part is optional; if omitted the
    model defaults to a pure classical regressor.
    """
    def __init__(
        self,
        quantum_circuit=None,
        quantum_estimator=None,
    ) -> None:
        super().__init__()
        self.feature_extractor = ClassicalFeatureExtractor()
        self.quantum_circuit = quantum_circuit
        self.quantum_estimator = quantum_estimator
        self.regressor = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        features = self.feature_extractor(x)  # (bsz, 4)

        # Quantum evaluation
        if self.quantum_circuit is not None and self.quantum_estimator is not None:
            # Assemble parameters: first 4 are inputs, last is a trainable weight
            params = np.concatenate(
                [features.detach().cpu().numpy(), np.zeros((features.shape[0], 1))],
                axis=1,
            )
            qnn = QiskitEstimatorQNN(
                circuit=self.quantum_circuit,
                observables=self.quantum_estimator.observable,
                input_params=self.quantum_estimator.input_params,
                weight_params=self.quantum_estimator.weight_params,
                estimator=self.quantum_estimator.estimator,
            )
            quantum_out = qnn.predict(params)
            quantum_out = torch.tensor(quantum_out, dtype=x.dtype, device=x.device)
        else:
            quantum_out = torch.zeros((x.shape[0], 1), device=x.device)

        # Combine classical and quantum outputs
        out = self.regressor(features) + quantum_out
        return out

__all__ = ["CombinedEstimatorQNN"]
