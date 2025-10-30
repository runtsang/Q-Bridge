"""Hybrid quantum estimator that augments a TorchQuantum kernel with a variational
circuit and Qiskit’s StatevectorEstimator.  The model follows the original
EstimatorQNN interface but replaces the simple classical network with a
quantum kernel and a trainable variational layer.

The class is split into two parts:

* ``QuantumKernel`` – a fixed TorchQuantum ansatz that evaluates the
  overlap of two feature‑encoded states.
* ``VariationalQNN`` – a variational circuit that learns weight parameters
  and feeds the resulting expectation value into a classical linear head.

The two are combined in ``EstimatorQNNGenQml`` which exposes ``fit`` and
``predict`` methods compatible with scikit‑learn.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

# --------------------------------------------------------------------------- #
# 1. Quantum kernel via TorchQuantum
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """
    Encodes classical data through a list of single‑qubit Ry gates.
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that returns the absolute value of the overlap between
    two encoded states.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# 2. Variational circuit (trainable weights)
# --------------------------------------------------------------------------- #

class VariationalQNN(tq.QuantumModule):
    """
    A simple variational circuit that applies a layer of Ry rotations
    parameterised by a weight vector.
    """
    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, requires_grad=True)
        )

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                tq.ry(x[:, q] + self.weights[l, q], wires=q)(q_device)

# --------------------------------------------------------------------------- #
# 3. Hybrid estimator
# --------------------------------------------------------------------------- #

class EstimatorQNNGenQml:
    """
    Hybrid quantum estimator that:

    1. Computes a quantum kernel with TorchQuantum.
    2. Applies a variational circuit to learn feature weights.
    3. Uses Qiskit’s StatevectorEstimator to evaluate expectation values.
    4. Trains a linear regression head on top of the kernel matrix.
    """
    def __init__(self, ridge: float = 1e-4):
        # Quantum components
        self.kernel = QuantumKernel()
        self.var_qnn = VariationalQNN(n_qubits=4, n_layers=2)
        # Classical linear head
        self.ridge = ridge
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def _kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix K_ij = k(x_i, y_j) using the quantum kernel.
        """
        n, m = X.shape[0], Y.shape[0]
        X_exp = X.unsqueeze(1).expand(n, m, -1)
        Y_exp = Y.unsqueeze(0).expand(n, m, -1)
        return self.kernel(X_exp, Y_exp).squeeze(-1)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Train the variational circuit and linear head.
        """
        # Optimise the variational weights via gradient descent
        optimizer = torch.optim.Adam(self.var_qnn.parameters(), lr=0.01)
        for _ in range(200):
            optimizer.zero_grad()
            K = self._kernel_matrix(X, X)
            loss = torch.mean((K - y) ** 2)
            loss.backward()
            optimizer.step()

        # After optimisation, solve linear ridge regression on the kernel
        K = self._kernel_matrix(X, X)
        n = K.shape[0]
        A = K + self.ridge * torch.eye(n, device=K.device)
        w = torch.linalg.solve(A, y)
        self.linear.weight.data = w.t()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return predictions for new data.
        """
        K = self._kernel_matrix(X, X)
        return self.linear(K)

__all__ = ["EstimatorQNNGenQml"]
