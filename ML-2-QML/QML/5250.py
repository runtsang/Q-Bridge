"""Hybrid quantum kernel regression implementation.

This module fuses the quantum kernel construction from
`QuantumKernelMethod.py` with the quantum regression model from
`QuantumRegression.py`.  The class ``HybridKernelRegression`` inherits
from :class:`torchquantum.QuantumModule` and exposes both a quantum
kernel and a regression head.  The kernel is evaluated by encoding
classical data into a quantum state and computing the overlap of two
states.  The regression head applies a variational circuit followed
by a measurement and a linear read‑out.

Key additions compared to the seeds:

*   The kernel width ``gamma`` is now a learnable parameter
    that scales the input angles before encoding.
*   The variational layer is built from a ``RandomLayer`` and
    trainable ``RX``/``RY`` gates, providing a richer feature space.
*   A ``predict`` method implements kernel ridge regression using
    the quantum kernel.
*   ``FastEstimator`` is implemented for quantum circuits using
    Qiskit’s Statevector simulator, enabling expectation value
    evaluation for many parameter sets.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

# ---------------------------------------------------------------------------

class HybridKernelRegression(tq.QuantumModule):
    """Quantum kernel regression with a learnable input scaling.

    Parameters
    ----------
    num_wires : int
        Number of qubits used to encode the input.
    n_wires : int, optional
        Number of qubits in the variational layer.  Defaults to ``num_wires``.
    """

    def __init__(self,
                 num_wires: int,
                 n_wires: int | None = None) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.n_wires = n_wires or num_wires
        # learnable scaling of the input before encoding
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        # encoder that maps a real vector to a superposition
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.variational = self._build_variational(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 1)

    # -----------------------------------------------------------------------
    # Variational layer

    def _build_variational(self, n_wires: int) -> tq.QuantumModule:
        class Variational(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

        return Variational()

    # -----------------------------------------------------------------------
    # Forward pass

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Map a batch of classical states to a regression output.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape ``(batch, num_wires)`` containing real values.
        """
        bsz = state_batch.shape[0]
        # scale the input with the learnable gamma
        scaled = self.gamma * state_batch
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, scaled)
        self.variational(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    # -----------------------------------------------------------------------
    # Quantum kernel

    class _QuantumKernel(tq.QuantumModule):
        """Computes the overlap between two encoded states."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
            bsz = x.shape[0]
            qdev.reset_states(bsz)
            self.encoder(qdev, x)
            self.encoder(qdev, y)
            # overlap is the absolute value of the first amplitude
            # after encoding both states (interference)
            self.measure(qdev)

        def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Return the overlap between two batches of states."""
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz)
            self.forward(qdev, x, y)
            return torch.abs(qdev.states.view(-1)[0])

    def kernel_matrix(self,
                      a: torch.Tensor,
                      b: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix between two datasets using the quantum kernel."""
        kernel = self._QuantumKernel(self.num_wires)
        return torch.stack([kernel.kernel_value(a[i:i+1], b) for i in range(a.shape[0])]).squeeze(1)

    # -----------------------------------------------------------------------
    # Prediction using kernel ridge regression

    def predict(self,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_test: torch.Tensor,
                reg_lambda: float = 1e-5) -> torch.Tensor:
        """Kernel ridge regression with the quantum kernel."""
        K = self.kernel_matrix(X_train, X_train)
        K += reg_lambda * torch.eye(K.shape[0], device=K.device)
        alpha = torch.linalg.solve(K, y_train)
        K_test = self.kernel_matrix(X_test, X_train)
        return (K_test @ alpha).squeeze(-1)

# ---------------------------------------------------------------------------

# Quantum dataset and utilities (from QuantumRegression.py)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic quantum dataset.

    The states are superpositions of |0...0> and |1...1> with random
    angles.  Labels are a simple trigonometric function of the angles.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a complex state vector and a real target."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ---------------------------------------------------------------------------

# Fast estimator for quantum circuits (adapted from FastBaseEstimator.py)

class FastQuantumEstimator:
    """Evaluate expectation values of quantum observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrised Qiskit circuit whose parameters will be bound.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = [
    "HybridKernelRegression",
    "generate_superposition_data",
    "RegressionDataset",
    "FastQuantumEstimator",
]
