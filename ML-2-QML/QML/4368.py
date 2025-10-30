"""Hybrid quantum estimator combining quanvolution, self‑attention and sampler circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuanvolutionFilter:
    """Quantum 2×2 patch encoder using a random two‑qubit layer."""

    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.encoder = QuantumCircuit(n_wires)
        for i in range(n_wires):
            self.encoder.ry(ParameterVector(f"ry_{i}", 1)[0], i)
        self.layer = QuantumCircuit(n_wires)
        self.layer.rz(ParameterVector("rz", 1)[0], 0)

    def bind(self, params: Sequence[float]) -> QuantumCircuit:
        qc = self.encoder.copy()
        for i, val in enumerate(params):
            qc.ry(val, i)
        qc += self.layer
        return qc


class QuantumSelfAttention:
    """Self‑attention block implemented as a small parameterised circuit."""

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def build(self, rotation: np.ndarray, entangle: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation[3 * i], i)
            qc.ry(rotation[3 * i + 1], i)
            qc.rz(rotation[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc


class QuantumSamplerQNN:
    """Parameterised sampler circuit producing a probability distribution."""

    def __init__(self) -> None:
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        self.circuit = qc

    def bind(self, inputs: Sequence[float], weights: Sequence[float]) -> QuantumCircuit:
        qc = self.circuit.copy()
        for i, val in enumerate(inputs):
            qc.ry(val, i)
        for i, val in enumerate(weights):
            qc.ry(val, i % 2)
        return qc


class HybridEstimator:
    """Quantum estimator that evaluates a composite circuit comprising
    quanvolution, self‑attention and sampler modules.

    The estimator exposes an ``evaluate`` method that returns expectation
    values for a list of Pauli operators.  It can run on a simulator
    backend and optionally add shot noise.
    """

    def __init__(
        self,
        *,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Assemble a composite circuit
        self.qfilter = QuanvolutionFilter()
        self.attn = QuantumSelfAttention()
        self.sampler = QuantumSamplerQNN()
        self.composite = self._build_composite()

    def _build_composite(self) -> QuantumCircuit:
        qc = QuantumCircuit()
        # Add quanvolution encoding
        qc += self.qfilter.bind([0.0] * self.qfilter.n_wires)
        # Add self‑attention block
        qc += self.attn.build(np.zeros(12), np.zeros(3))
        # Add sampler
        qc += self.sampler.circuit
        return qc

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            # Bind parameters to the composite circuit
            bound = self.composite.copy()
            for i, val in enumerate(params):
                bound.ry(val, i % bound.num_qubits)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def sample_counts(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[dict]:
        """Return measurement counts for each parameter set on the sampler circuit."""
        counts: List[dict] = []
        for params in parameter_sets:
            qc = self.sampler.bind(params[:2], params[2:6])
            job = execute(qc, self.backend, shots=self.shots)
            counts.append(job.result().get_counts(qc))
        return counts
