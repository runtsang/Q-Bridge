"""Hybrid QCNN model combining quantum feature extraction with a classical classifier.

This module builds on the original QCNNModel by adding an optional quantum
feature map (ZFeatureMap) that is evaluated with a state‑vector simulator.
When ``use_quantum_feature_map`` is ``True`` the input data is first
transformed into a quantum feature vector before the classical network.
The resulting architecture is fully differentiable and can be trained
with any PyTorch optimiser.
"""

import torch
from torch import nn
from typing import Optional

# Optional quantum feature map; imported only when needed to avoid heavy imports
try:
    from qiskit.circuit.library import ZFeatureMap
    from qiskit.quantum_info import Statevector
    from qiskit import Aer
except ImportError:
    ZFeatureMap = None
    Statevector = None
    Aer = None


class QCNNHybrid(nn.Module):
    """Hybrid QCNN: optional quantum feature extraction followed by a
    fully‑connected classifier.

    Parameters
    ----------
    input_dim : int
        Dimension of the raw input data.
    use_quantum_feature_map : bool, default=False
        If True, the input is passed through a ZFeatureMap evaluated on a
        state‑vector simulator before the classical layers.
    quantum_feature_dim : int, default=8
        Dimension of the quantum feature vector produced by the feature map.
    """

    def __init__(
        self,
        input_dim: int = 8,
        use_quantum_feature_map: bool = False,
        quantum_feature_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_quantum = use_quantum_feature_map

        if self.use_quantum:
            if ZFeatureMap is None:
                raise ImportError(
                    "qiskit is required for quantum feature extraction but "
                    "was not importable."
                )
            self.feature_map = ZFeatureMap(num_qubits=quantum_feature_dim)
            self.quantum_dim = quantum_feature_dim
            self.classifier_input_dim = quantum_feature_dim
        else:
            self.classifier_input_dim = input_dim

        # Classical classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def _quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the ZFeatureMap on a state‑vector simulator for each sample
        in ``x`` and return the expectation value of Pauli‑Z on each qubit.
        """
        if Statevector is None or Aer is None:
            raise ImportError(
                "qiskit is required for quantum feature extraction but "
                "was not importable."
            )
        backend = Aer.get_backend("statevector_simulator")
        features = []
        for sample in x:
            qc = self.feature_map.assign_parameters(sample.numpy(), inplace=False)
            result = backend.run(qc).result()
            state = Statevector(result.get_statevector())
            # expectation values of Z on each qubit
            exp_vals = [state.expectation_value(f"Z{i}") for i in range(self.quantum_dim)]
            features.append(exp_vals)
        return torch.tensor(features, dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid model."""
        if self.use_quantum:
            x = self._quantum_features(x)
        return torch.sigmoid(self.classifier(x))


def QCNN() -> QCNNHybrid:
    """Factory returning a hybrid QCNN with default settings."""
    return QCNNHybrid()


__all__ = ["QCNN", "QCNNHybrid"]
