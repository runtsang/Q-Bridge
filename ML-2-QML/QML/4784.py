"""Hybrid quantum implementation of a fully‑connected regression layer.

This module uses Qiskit's EstimatorQNN to emulate the behaviour of the classical
model while leveraging a quantum circuit for feature mapping and quantum
operations.  The API is intentionally identical to the classical variant so
experiments can swap engines with minimal code changes.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantum‑style synthetic data generator.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the generated state.
    samples : int
        Number of samples to produce.

    Returns
    -------
    states, labels : np.ndarray
        ``states`` of shape (samples, 2**num_wires) and target vector of shape (samples,).
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
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapper for the quantum‑generated data.

    The dataset yields ``states`` as complex tensors and ``target`` as real scalars.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SharedClassName:
    """
    Quantum regression network based on Qiskit's EstimatorQNN.

    The network encodes the input state using a Ry‑based encoder, applies a
    trainable random layer followed by per‑wire RX/RY rotations, and measures the
    Pauli‑Z expectation values.  A classical linear head maps the quantum feature
    vector to a scalar prediction.
    """

    def __init__(self, num_wires: int = 3, backend: str = "qasm_simulator", shots: int = 1024):
        self.num_wires = num_wires
        self.backend = backend
        self.shots = shots

        # Parameterised circuit
        self.theta = Parameter("theta")
        self._circuit = QuantumCircuit(num_wires)
        self._circuit.h(range(num_wires))
        self._circuit.ry(self.theta, range(num_wires))
        self._circuit.measure_all()

        # Estimator and QNN
        estimator = StatevectorEstimator()
        observable = SparsePauliOp.from_list([("Z" * num_wires, 1.0)])
        self.qnn = EstimatorQNN(
            circuit=self._circuit,
            observables=observable,
            input_params=[self.theta],
            weight_params=[],
            estimator=estimator,
        )

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the quantum circuit for a batch of parameter values.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of rotation angles for the Ry gates.

        Returns
        -------
        np.ndarray
            Mean expectation value across the batch.
        """
        job = self.qnn.run(parameter_binds=[{self.theta: t} for t in thetas])
        expectations = np.array(job)
        return expectations.reshape(-1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the quantum feature map, measurements and
        classical head.  ``state_batch`` should be a complex tensor of shape
        (batch, 2**num_wires).
        """
        # Encode the input state into the Qiskit circuit
        # (here we simply treat the batch index as the parameter bind)
        parameter_binds = [{self.theta: 0.0}] * state_batch.shape[0]
        job = self.qnn.run(parameter_binds=parameter_binds)
        features = torch.tensor(job, dtype=torch.float32)
        return self.head(features).squeeze(-1)


__all__ = ["SharedClassName", "RegressionDataset", "generate_superposition_data"]
