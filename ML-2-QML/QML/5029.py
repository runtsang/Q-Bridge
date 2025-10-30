"""Quantum implementation of UnifiedFCL using TorchQuantum and Qiskit for evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Sequence, List
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qiskit


class UnifiedFCL(tq.QuantumModule):
    """
    Quantum hybrid model mirroring the classical UnifiedFCL.
    Encodes a 2**n_wires state, applies a random circuit and trainable RX/RY gates,
    measures all qubits, and maps the expectation vector to a scalar target.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps a real‑valued vector onto a 2‑qubit state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

        # Build a template Qiskit circuit for evaluation
        self._base_circuit = QuantumCircuit(num_wires)
        self._params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(num_wires)]
        for i in range(num_wires):
            self._base_circuit.ry(self._params[i], i)
        # random layer is omitted in Qiskit evaluation for simplicity

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: (batch, 2**num_wires) complex tensor representing a pure state.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each set of parameters over the given observables.
        Parameters are bound to the circuit's trainable gates.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._circuit_with_params(values)
            sv = Statevector.from_instruction(circ)
            row = [sv.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _circuit_with_params(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Return a Qiskit circuit with parameters bound to the module."""
        circ = self._base_circuit.copy()
        binding = {p: v for p, v in zip(self._params, param_values)}
        circ.bind_parameters(binding)
        circ.measure_all()
        return circ


__all__ = ["UnifiedFCL"]
