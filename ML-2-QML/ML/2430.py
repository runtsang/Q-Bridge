"""Hybrid classical‑quantum fully‑connected layer.

This module merges the classical feed‑forward architecture from
`QuantumClassifierModel.build_classifier_circuit` with a
parameterized quantum circuit from the quantum counterpart.
The resulting class can be used as a drop‑in replacement for the
original `FCL` while exposing both classical and quantum
inference paths.

Typical usage:
    >>> from FCL__gen237 import FCL
    >>> model = FCL()
    >>> # Classical inference
    >>> logits = model.run_classical([0.5, 0.2])
    >>> # Quantum inference
    >>> thetas = [0.1] * (model.n_features * model.depth)
    >>> expectations = model.run_quantum(thetas, [0.5, 0.2])
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
from torch import nn

# Quantum imports
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that supports both classical and quantum
    evaluation modes.

    Parameters
    ----------
    n_features : int
        Number of input features / qubits.
    depth : int
        Depth of the feed‑forward network / quantum ansatz.
    use_quantum : bool, default=True
        If True, :meth:`run` will execute the quantum circuit.
        If False, the classical network is used.
    backend : qiskit.providers.Backend, optional
        Quantum backend; defaults to Aer qasm_simulator.
    shots : int, default=1024
        Number of shots for the quantum simulation.
    """

    def __init__(
        self,
        n_features: int = 1,
        depth: int = 1,
        use_quantum: bool = True,
        backend=None,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.use_quantum = use_quantum
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # ---------- Classical network ----------
        layers: List[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, n_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = n_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.classical_net = nn.Sequential(*layers)

        # ---------- Quantum circuit ----------
        # Encoding and variational parameters
        self.encoding = ParameterVector("x", n_features)
        self.weights = ParameterVector("theta", n_features * depth)

        self.qc = QuantumCircuit(n_features)
        # Data encoding
        for param, qubit in zip(self.encoding, range(n_features)):
            self.qc.rx(param, qubit)

        # Ansatz layers
        idx = 0
        for _ in range(depth):
            for qubit in range(n_features):
                self.qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(n_features - 1):
                self.qc.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (n_features - i - 1))
            for i in range(n_features)
        ]

        self.qc.measure_all()

    # ---------------- Classical API ----------------
    def run_classical(self, input_data: Iterable[float]) -> np.ndarray:
        """Run the classical feed‑forward network."""
        with torch.no_grad():
            x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            logits = self.classical_net(x)
            return logits.detach().numpy().squeeze()

    # ---------------- Quantum API ----------------
    def run_quantum(self, thetas: Iterable[float], input_data: Iterable[float]) -> np.ndarray:
        """Run the quantum circuit and return expectation values."""
        # Bind encoding and weights
        param_binds = [
            {self.encoding[i]: val for i, val in enumerate(input_data)},
            {self.weights[i]: val for i, val in enumerate(thetas)},
        ]
        job = qiskit.execute(
            self.qc,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.qc)

        expectation = np.zeros(len(self.observables))
        for state, cnt in counts.items():
            prob = cnt / self.shots
            bits = [int(b) for b in state[::-1]]  # Qiskit uses little‑endian
            for idx, _ in enumerate(self.observables):
                expectation[idx] += prob * (1 if bits[idx] == 0 else -1)
        return expectation

    # ---------------- Unified API ----------------
    def run(self, thetas: Iterable[float], input_data: Iterable[float]) -> np.ndarray:
        """Dispatch to the selected evaluation mode."""
        if self.use_quantum:
            return self.run_quantum(thetas, input_data)
        return self.run_classical(input_data)


def FCL() -> HybridFCL:
    """Return a ready‑to‑use hybrid fully‑connected layer."""
    return HybridFCL()


__all__ = ["HybridFCL", "FCL"]
