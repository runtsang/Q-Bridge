"""Hybrid Estimator with classical feed-forward and quantum self‑attention feature extractor.

The module implements a PyTorch model that augments a conventional neural network with a
quantum self‑attention block.  The quantum block is a single‑qubit circuit whose rotation
angles are functions of the input features and whose weight parameter controls the
entanglement‑like rotation.  The expectation value of Pauli‑Y on the final state is
treated as a learned feature and concatenated with the raw inputs before passing them
through the classical feed‑forward layers.

The scaling paradigm is `combination`: classical and quantum components cooperate
rather than replace one another.
"""

import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter


class HybridEstimatorQNN(nn.Module):
    """
    Classical feed‑forward regressor with a quantum self‑attention feature extractor.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : list[int]
        Sizes of the hidden layers in the classical network.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]

        # Classical feed‑forward backbone
        layers = []
        prev_dim = input_dim + 1  # +1 for the quantum feature
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*layers)

        # Parameters for the quantum circuit
        self.weight_param = nn.Parameter(torch.randn(1))

        # Quantum backend
        self.backend = Aer.get_backend("statevector_simulator")

    def _quantum_feature(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation value of Pauli‑Y on a single‑qubit circuit.

        The rotation angles are linear functions of the two input features.
        """
        batch_size = inputs.shape[0]
        features = []

        for i in range(batch_size):
            x = inputs[i].detach().cpu().numpy()
            # Rotation angles derived from the input
            rot1 = x[0]
            rot2 = x[1]
            # Weight parameter controls an additional rotation
            w = self.weight_param.detach().cpu().item()

            qc = QuantumCircuit(1)
            qc.h(0)
            qc.ry(rot1, 0)
            qc.rx(rot2, 0)
            qc.rx(w, 0)

            job = execute(qc, self.backend)
            state = job.result().get_statevector(qc)
            # Pauli‑Y expectation
            y_op = np.array([[0, -1j], [1j, 0]])
            exp = np.real(state.conj().T @ (y_op @ state))
            features.append(exp)

        return torch.tensor(features, dtype=torch.float32).unsqueeze(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: prepend the quantum feature to the raw inputs and feed them
        through the classical network.
        """
        q_feat = self._quantum_feature(inputs)
        augmented = torch.cat([inputs, q_feat], dim=1)
        return self.fc(augmented)


__all__ = ["HybridEstimatorQNN"]
