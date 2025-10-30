"""Quantum kernel estimator that replaces the classical RBF kernel with a variational circuit
and trains a Qiskit EstimatorQNN on the resulting Gram matrix.

The public API mirrors the classical implementation so that both modules can be swapped
in a single experiment script.  The quantum kernel is a fixed 4‑qubit ansatz; the QNN
is a trivially parameterised circuit that learns a mapping from kernel values to target
labels.

Typical usage:

    from QuantumKernelMethod__gen020 import QuantumKernelEstimator
    X_train, y_train =...
    model = QuantumKernelEstimator(gamma=0.5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
#  Quantum kernel – adapted from the seed
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a list of single‑qubit rotations."""
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


class Kernel(tq.QuantumModule):
    """Fixed 4‑qubit ansatz used to compute quantum kernel values."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


# --------------------------------------------------------------------------- #
#  Qiskit EstimatorQNN – adapted from the seed
# --------------------------------------------------------------------------- #

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

def _build_qnn() -> EstimatorQNN:
    """Construct a minimal 1‑qubit EstimatorQNN that will be trained on kernel features."""
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.ry(params[0], 0)
    qc.ry(params[1], 0)
    observable = SparsePauliOp.from_list([("I", 1)])
    estimator = Estimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )

# --------------------------------------------------------------------------- #
#  High‑level estimator
# --------------------------------------------------------------------------- #

class QuantumKernelEstimator:
    """
    Quantum‑kernel based estimator that trains a Qiskit EstimatorQNN on the Gram matrix.

    Parameters
    ----------
    gamma : float
        Width of the kernel (passed to the quantum kernel).
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        self.q_kernel = Kernel()
        self.qnn = _build_qnn()

    # --------------------------------------------------------------------- #
    #  Kernel helpers
    # --------------------------------------------------------------------- #
    def _kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between X and Y using the quantum kernel."""
        return np.array([[self.q_kernel(x, y).item() for y in Y] for x in X])

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def fit(self, X: Sequence[torch.Tensor], y: Iterable[float]) -> None:
        """
        Train the EstimatorQNN on the kernel matrix.

        The kernel matrix is treated as a feature matrix; the QNN learns a mapping
        from those features to the target labels.
        """
        X = torch.stack(list(X))
        K = self._kernel_matrix(X, X)
        self.qnn.fit(K, np.asarray(list(y)))

    def predict(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        """Predict on new data using the trained QNN."""
        X = torch.stack(list(X))
        K = self._kernel_matrix(X, X)
        return self.qnn.predict(K)

    def __repr__(self) -> str:
        return f"<QuantumKernelEstimator gamma={self.gamma} qnn={self.qnn!r}>"

__all__ = ["QuantumKernelEstimator"]
