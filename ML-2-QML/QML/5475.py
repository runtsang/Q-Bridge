"""Quantum hybrid self‑attention model.

This module implements the quantum side of the pipeline.  It
exposes the same public API as the classical variant so that the
two can be interchanged.  The attention block, classifier ansatz,
and kernel are all built with Qiskit or TorchQuantum, and the
FastBaseEstimator evaluates expectation values on a Statevector
simulator.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

from.QuantumClassifierModel import build_classifier_circuit as build_q_cls
from.QuantumKernelMethod import Kernel as QuantumKernel
from.FastBaseEstimator import FastBaseEstimator


class QuantumSelfAttention:
    """Variational self‑attention implemented with Qiskit."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class HybridSelfAttentionModel:
    """Quantum‑only self‑attention pipeline."""

    def __init__(
        self,
        n_qubits: int,
        embed_dim: int,
        depth: int,
        backend=None,
    ) -> None:
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Attention block
        self.attention = QuantumSelfAttention(n_qubits)

        # Classifier circuit
        (
            self.classifier_circuit,
            self.cls_encoding,
            self.cls_weights,
            self.cls_obs,
        ) = build_q_cls(n_qubits, depth)

        # Quantum kernel
        self.kernel = QuantumKernel()

        # Estimator
        self.estimator = FastBaseEstimator(self.classifier_circuit)

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, np.ndarray]:
        """Execute the attention circuit and return the probability distribution."""
        counts = self.attention.run(
            self.backend,
            rotation_params,
            entangle_params,
            shots=shots,
        )
        probs = np.array(
            [counts.get(bit, 0) for bit in sorted(counts)]
        )
        return {"probabilities": probs / probs.sum() if probs.sum() else probs}

    def evaluate_classifier(
        self,
        parameters: Sequence[float],
    ) -> np.ndarray:
        """Return the statevector after the classifier ansatz."""
        bound = self.classifier_circuit.assign_parameters(
            dict(zip(self.classifier_circuit.parameters, parameters))
        )
        sv = Statevector.from_instruction(bound)
        return sv.data

    def kernel_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute the quantum kernel matrix between two datasets."""
        kernel = self.kernel
        return np.array(
            [
                [
                    kernel(
                        torch.tensor(x_i, dtype=torch.float32),
                        torch.tensor(y_j, dtype=torch.float32),
                    ).item()
                    for y_j in y
                ]
                for x_i in x
            ]
        )

    def estimate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[SparsePauliOp],
    ) -> List[List[complex]]:
        """Delegate to the Qiskit FastBaseEstimator."""
        return self.estimator.evaluate(observables, parameter_sets)
