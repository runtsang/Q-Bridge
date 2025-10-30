"""Quantum‑augmented Quanvolution classifier.

This variant implements the entire pipeline with TorchQuantum modules, while
mirroring the classical interface.  It uses:

* A quantum filter (QuanvolutionFilterQuantum) identical to the classical one.
* A variational classifier circuit built by `build_classifier_circuit` from the
  quantum reference.
* A FastBaseEstimator analogue that evaluates state‑vector expectation values
  and optionally adds Gaussian shot noise.

The class name and public API match the classical version, making it a drop‑in
replacement for experiments that wish to swap between classical and quantum
back‑ends.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

# --------------------------------------------------------------------------- #
# Quantum filter (TorchQuantum)
# --------------------------------------------------------------------------- #
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum 2×2 patch encoder using a random two‑qubit layer."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:  # type: ignore[override]
        # x: (batch, 4) patches
        q_device.reset_states(x.shape[0])
        for idx, patch in enumerate(x.split(1, dim=1)):
            self.encoder(q_device, patch.squeeze(1))
            self.q_layer(q_device)
            measurement = self.measure(q_device)
            if idx == 0:
                out = measurement.view(-1, 4)
            else:
                out = torch.cat([out, measurement.view(-1, 4)], dim=0)
        self.output = out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        q_device = tq.QuantumDevice(self.n_wires, bsz=x.shape[0])
        self.forward(q_device, x)
        return self.output


# --------------------------------------------------------------------------- #
# Estimator utilities (quantum)
# --------------------------------------------------------------------------- #
class FastBaseEstimatorQuantum:
    """
    Evaluate expectation values of observables for a parametrized circuit.
    """

    def __init__(self, circuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]):
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimatorQuantum(FastBaseEstimatorQuantum):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean.real, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Classifier factory (quantum)
# --------------------------------------------------------------------------- #
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Main quantum model
# --------------------------------------------------------------------------- #
class Quanvolution__gen319(tq.QuantumModule):
    """
    Quantum‑augmented quanvolution classifier.

    Parameters
    ----------
    depth : int
        Depth of the variational classifier circuit.
    """

    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterQuantum()
        num_qubits = 4 * 14 * 14  # flattening after encoding
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:  # type: ignore[override]
        # Encode image patches
        self.filter(q_device, x)
        # Append classifier weights
        self.circuit.assign_parameters(q_device.states.view(-1, self.weights.shape[0]), inplace=True)
        # Apply classifier circuit
        q_device.apply_circuit(self.circuit)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Create a new device for every forward
        q_device = tq.QuantumDevice(self.filter.n_wires, bsz=x.shape[0])
        self.forward(q_device, x)
        return q_device.states.view(-1, self.circuit.num_qubits).log_prob()

    # Estimator API ---------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        estimator = FastEstimatorQuantum(self.circuit) if shots is not None else FastBaseEstimatorQuantum(self.circuit)
        return estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["Quanvolution__gen319"]
