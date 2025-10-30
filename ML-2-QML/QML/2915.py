"""Quantum attention‑classifier hybrid.

Combines a variational attention ansatz with a quantum classifier circuit
from the seeds.  The class exposes a single `run` method that accepts a
backend, parameter vectors, and returns measurement counts or expectation
values.  It is designed for side‑by‑side comparison with the classical
counterpart above.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Import the quantum classifier factory from the seed.
from build_classifier_circuit import build_classifier_circuit

class QuantumAttentionClassifier:
    """Variational attention followed by a quantum classifier."""

    def __init__(self, num_qubits: int, depth: int):
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits used for both attention and classifier.
        depth : int
            Depth of the variational layers in both blocks.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.attention_circuit, self.enc_params, self.att_params, _ = build_attention_circuit(
            num_qubits, depth
        )
        self.classifier_circuit, self.cls_enc, self.cls_params, self.observables = build_classifier_circuit(
            num_qubits, depth
        )

    def _build_full_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Concatenate attention and classifier circuits."""
        full = QuantumCircuit(self.num_qubits)
        # Attention block
        full.compose(
            self.attention_circuit.bind_parameters(
                dict(zip(self.enc_params, rotation_params))
            ),
            inplace=True,
        )
        # Classifier block
        full.compose(
            self.classifier_circuit.bind_parameters(
                dict(zip(self.cls_enc, entangle_params))
            ),
            inplace=True,
        )
        return full

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> List[Tuple[str, float]]:
        """
        Execute the full circuit and return expectation values.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Backend to execute the circuit on.
        rotation_params : np.ndarray
            Parameters for the attention encoding.
        entangle_params : np.ndarray
            Parameters for the attention variational layers.
        shots : int, optional
            Number of shots for measurement.

        Returns
        -------
        List[Tuple[str, float]]
            List of (observable, expectation_value).
        """
        circuit = self._build_full_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Convert counts to expectation values for each observable
        exp_vals = []
        for obs in self.observables:
            exp = result.get_expectation_value(obs, circuit)
            exp_vals.append((str(obs), exp))
        return exp_vals

def build_attention_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a variational attention ansatz (mirrors the SelfAttention seed)."""
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

__all__ = ["QuantumAttentionClassifier"]
