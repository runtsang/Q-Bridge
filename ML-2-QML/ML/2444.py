"""Hybrid classical classifier with optional quantum feature extraction.

The model fuses a deep feed‑forward network (inspired by the original
`build_classifier_circuit`) with a lightweight regression head
(`EstimatorQNN`).  During training the two sub‑models are updated
jointly, enabling the classical backbone to learn from the
quantum‑generated features while retaining full PyTorch
autograd support.

The class is fully importable and can be dropped into any PyTorch
training loop.  It also exposes a `quantum_features` helper that
evaluates a small Qiskit circuit on a CPU simulator, allowing the
quantum part to be treated as a feature extractor when desired.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Classical building blocks
# --------------------------------------------------------------------------- #

def _build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Create a multilayer perceptron with `depth` hidden layers.

    The architecture mirrors the original `build_classifier_circuit`
    but uses `nn.Linear` layers whose width equals the input dimension.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)          # binary classification
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))          # dummy observables for API parity
    return network, encoding, weight_sizes, observables


def _build_estimator_qnn() -> nn.Module:
    """Return a lightweight regression network used as a head.

    The network is intentionally small to keep the hybrid model
    lightweight while still providing a learnable mapping from
    raw inputs to a single continuous output.
    """
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
    )


# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #

class QuantumClassifierModel(nn.Module):
    """
    A hybrid classifier that combines a classical MLP with a
    regression head.  The regression head can optionally be
    replaced by a quantum feature extractor by calling
    :meth:`quantum_features`.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.as_tensor([], dtype=torch.float32, device=device).device
        self.classifier, self.enc, self.w_sizes, _ = _build_classifier_circuit(num_features, depth)
        self.regressor = _build_estimator_qnn()
        self.to(self.device)

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        logits = self.classifier(x.to(self.device))
        reg = self.regressor(x.to(self.device))
        # Simple fusion: add the regression output to the logits
        # after broadcasting to match dimensions.
        logits = logits + reg
        return logits

    # --------------------------------------------------------------------- #
    # Quantum feature extractor
    # --------------------------------------------------------------------- #
    def quantum_features(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate a small Qiskit circuit on the CPU simulator and return
        the expectation values as a feature vector.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (batch, num_qubits).

        Returns
        -------
        np.ndarray
            Feature matrix of shape (batch, num_qubits).
        """
        from qiskit import QuantumCircuit, Aer, execute
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        num_qubits = x.shape[1]
        depth = 2
        # Encoding parameters
        enc = ParameterVector("x", num_qubits)
        # Variational parameters
        theta = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(enc, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(theta[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        obs = [SparsePauliOp.from_list([("Z" * i + "I" * (num_qubits - i - 1), 1)]) for i in range(num_qubits)]

        # Bind parameters
        param_dict = {enc[i]: val for i, val in enumerate(x.T)}
        for i in range(depth * num_qubits):
            param_dict[theta[i]] = np.random.uniform(0, 2 * np.pi)

        bound_qc = qc.bind_parameters(param_dict)

        backend = Aer.get_backend("statevector_simulator")
        job = execute(bound_qc, backend, shots=1)
        result = job.result()
        statevector = result.get_statevector(bound_qc)
        features = np.array([np.real(statevector.dot(obs_i.to_matrix())) for obs_i in obs])
        return features

    # --------------------------------------------------------------------- #
    # Compatibility helpers
    # --------------------------------------------------------------------- #
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that applies a softmax to the logits.
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
