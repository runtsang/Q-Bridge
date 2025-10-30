"""Utility helpers for building classifiers and convolutional filters used by the hybrid estimator."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


def build_classifier_circuit(
    num_units: int,
    depth: int,
    *,
    quantum: bool = False,
) -> Tuple[Any, Iterable, Iterable, List[Any]]:
    """Return a classifier model, encoding parameters, weight vector sizes, and observables.

    Parameters
    ----------
    num_units
        Number of features (classical) or qubits (quantum).
    depth
        Number of hidden layers / ansatz layers.
    quantum
        If ``True`` a Qiskit circuit is returned; otherwise a PyTorch network.
    """
    if quantum:
        encoding = ParameterVector("x", num_units)
        weights = ParameterVector("theta", num_units * depth)
        circuit = QuantumCircuit(num_units)

        # Feature map
        for param, qubit in zip(encoding, range(num_units)):
            circuit.rx(param, qubit)

        # Variational layers
        w_idx = 0
        for _ in range(depth):
            for qubit in range(num_units):
                circuit.ry(weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(num_units - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_units - i - 1))
            for i in range(num_units)
        ]
        return circuit, list(encoding), list(weights), observables

    # Classical neural network
    layers = []
    in_dim = num_units
    encoding = list(range(num_units))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_units)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_units
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


def conv_factory(
    kernel_size: int = 2,
    threshold: float = 0.0,
    *,
    shots: int = 100,
    backend_name: str = "qasm_simulator",
) -> Tuple["ConvFilter", "QuanvCircuit"]:
    """Return a classical ConvFilter and a quantum QuanvCircuit for data‑dependent filtering.

    The classical filter uses a 2‑D convolution with a sigmoid activation.
    The quantum filter encodes the flattened kernel into rotation angles and
    runs a random circuit, measuring all qubits to produce a probability of
    observing |1>.
    """

    class ConvFilter(nn.Module):
        """Deterministic 2‑D convolution filter."""

        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: np.ndarray) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    class QuanvCircuit:
        """Quantum filter that maps data to rotation angles and measures."""

        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = QuantumCircuit(self.n_qubits)
            self.theta = [ParameterVector(f"theta{i}") for i in range(self.n_qubits)][0]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += QuantumCircuit.random(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    backend = Aer.get_backend(backend_name)
    return ConvFilter(kernel_size, threshold), QuanvCircuit(kernel_size, backend, shots, threshold)


__all__ = ["build_classifier_circuit", "conv_factory"]
