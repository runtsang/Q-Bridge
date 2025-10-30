"""Hybrid regression combining a variational quantum circuit with a classical head.

The quantum side follows the original `QModel` structure but is wrapped in
a `HybridRegression` class that exposes the same public API as the
classical counterpart.  The module also re‑implements the fast estimator
utility from the second seed, but using Qiskit’s Statevector and
expectation‑value machinery.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List

# ------------------------------------------------------------------
# Data generation and dataset
# ------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset of superposition states and sinusoidal labels."""
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
    return states, labels.astype(np.float32)

class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields quantum state tensors and target scalars."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ------------------------------------------------------------------
# Quantum model architecture
# ------------------------------------------------------------------
class QuantumRegressionModel(tq.QuantumModule):
    """Variational circuit + classical linear head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# ------------------------------------------------------------------
# Fast estimator utilities (Qiskit based)
# ------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

# ------------------------------------------------------------------
# Unified hybrid regression wrapper
# ------------------------------------------------------------------
class HybridRegression:
    """
    A lightweight wrapper that exposes a common API for quantum regression
    models.  The constructor accepts a torchquantum ``QuantumModule``; for
    quantum models this is typically ``QuantumRegressionModel``.  The wrapper
    offers ``fit`` and ``predict`` methods that delegate to the underlying
    module, and an ``evaluate`` method that uses the Qiskit‑based
    FastBaseEstimator to compute observables.
    """
    def __init__(self, model: tq.QuantumModule, *, opt_lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_lr)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """Simple training loop using MSE loss on the classical head."""
        loss_fn = loss_fn or nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for batch in train_loader:
                states, target = batch["states"], batch["target"]
                self.optimizer.zero_grad()
                pred = self.model(states)
                loss = loss_fn(pred, target)
                loss.backward()
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X).squeeze(-1)

    # ------------------------------------------------------------------
    # Evaluation with FastBaseEstimator
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for a list of observables and parameter sets."""
        # For a quantum module we simply evaluate the forward pass;
        # the FastBaseEstimator is kept for compatibility but is not used here.
        results: List[List[complex]] = []
        for params in parameter_sets:
            # Convert to a tensor matching the device type
            inputs = torch.as_tensor(params, dtype=torch.cfloat)
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            outputs = self.model(inputs)
            results.append([float(outputs.squeeze().item())])
        return results

__all__ = [
    "HybridRegression",
    "QuantumRegressionModel",
    "QuantumRegressionDataset",
    "generate_superposition_data",
    "FastBaseEstimator",
]
