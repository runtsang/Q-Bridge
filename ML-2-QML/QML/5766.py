"""Quantum implementation of QuantumNATHybrid using Pennylane. Encodes classical features into a variational circuit."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import List, Callable
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumNATHybrid:
    """Quantum‑only variant of the hybrid model. Uses a variational circuit to process classical embeddings."""

    class QLayer:
        """Variational block with random rotation layers and trainable parameters."""

        def __init__(self, n_wires: int, n_layers: int = 2):
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.params = nn.Parameter(
                torch.randn(n_layers, n_wires, 3)  # RX, RY, RZ per wire
            )

        def __call__(self, qdev=None):
            for layer in range(self.n_layers):
                for wire in range(self.n_wires):
                    qml.RX(self.params[layer, wire, 0], wires=wire)
                    qml.RY(self.params[layer, wire, 1], wires=wire)
                    qml.RZ(self.params[layer, wire, 2], wires=wire)
                # Entangling layer
                for wire in range(self.n_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])

    def __init__(self, n_qubits: int = 4, device_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits, shots=1024)
        self.encoder = qml.templates.AngleEmbedding
        self.q_layer = self.QLayer(n_qubits)
        self.measure = qml.expval(qml.PauliZ(0))
        self.norm = nn.BatchNorm1d(1)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor):
            # Classical encoding
            self.encoder(inputs, wires=range(n_qubits))
            # Variational block
            self.q_layer(self.dev)
            # Measurement
            return self.measure

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Global average pool to 4 features per sample
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        outputs = []
        for i in range(bsz):
            out = self.circuit(pooled[i])
            outputs.append(out)
        out_tensor = torch.stack(outputs)
        return self.norm(out_tensor)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Expectation values for a set of parameterised circuits and observables."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind_params(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind_params(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= self.n_qubits * 2:  # placeholder assumption
            raise ValueError("Parameter count mismatch for bound circuit.")
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(param_values):
            wire = i % self.n_qubits
            qc.rx(val, wire)
        return qc


__all__ = ["QuantumNATHybrid"]
