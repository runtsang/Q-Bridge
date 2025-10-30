"""Quantum self‑attention using a variational circuit with amplitude embedding."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import AmplitudeEmbedding
from qiskit.quantum_info import Statevector
from typing import Iterable, Tuple

class SelfAttentionEnhancedQ:
    """Variational self‑attention circuit with amplitude encoding.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match the embedding dimension).
    n_layers : int, default 1
        Number of variational layers.
    entangler_map : Iterable[Tuple[int, int]], default consecutive pairs
        Defines which qubits are entangled in each layer via CX gates.
    """
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        entangler_map: Iterable[Tuple[int, int]] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler_map = list(entangler_map) if entangler_map else [(i, i + 1) for i in range(n_qubits - 1)]

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> QuantumCircuit:
        """Construct the variational circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(n_layers, n_qubits, 3)`` – angles for RY, RZ, RX.
        entangle_params : np.ndarray
            Shape ``(n_layers, len(entangler_map))`` – angles for CRX.
        inputs : np.ndarray
            1‑D array of length ``2 ** n_qubits`` to be amplitude‑encoded.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Amplitude embedding
        amplitude_circ = AmplitudeEmbedding(
            inputs,
            qubits=list(range(self.n_qubits)),
            normalize=True,
            pad_with=0.0,
        )
        circuit.compose(amplitude_circ, inplace=True)

        for layer in range(self.n_layers):
            # Parameterized single‑qubit rotations
            for q in range(self.n_qubits):
                circuit.ry(rotation_params[layer, q, 0], q)
                circuit.rz(rotation_params[layer, q, 1], q)
                circuit.rx(rotation_params[layer, q, 2], q)
            # Entangling layer
            for idx, (q1, q2) in enumerate(self.entangler_map):
                circuit.crx(entangle_params[layer, idx], q1, q2)

        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        backend: qiskit.providers.Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
        return_expectation: bool = False,
    ):
        """Execute the circuit on the given backend.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Target backend.
        rotation_params : np.ndarray
            Shape ``(n_layers, n_qubits, 3)``.
        entangle_params : np.ndarray
            Shape ``(n_layers, len(entangler_map))``.
        inputs : np.ndarray
            Amplitude‑encoded input vector.
        shots : int, default 1024
            Number of shots for sampling.
        return_expectation : bool, default False
            If True, returns the sum of Z‑expectation values over all qubits
            instead of measurement counts.

        Returns
        -------
        dict or float
            Measurement counts if ``return_expectation`` is False,
            otherwise the scalar expectation value.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result()
        if return_expectation:
            state = Statevector.from_instruction(circuit)
            exp = 0.0
            for q in range(self.n_qubits):
                pauli_str = "Z" * q + "I" * (self.n_qubits - q - 1)
                exp += state.expectation_value(pauli_str)
            return exp
        else:
            return result.get_counts(circuit)

    def _expectation_value(self, circuit: QuantumCircuit) -> float:
        """Compute the sum of Z‑expectation values for all qubits."""
        state = Statevector.from_instruction(circuit)
        exp = 0.0
        for q in range(self.n_qubits):
            pauli_str = "Z" * q + "I" * (self.n_qubits - q - 1)
            exp += state.expectation_value(pauli_str)
        return exp

    def gradient(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shift: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate gradients via the parameter‑shift rule.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(n_layers, n_qubits, 3)``.
        entangle_params : np.ndarray
            Shape ``(n_layers, len(entangler_map))``.
        inputs : np.ndarray
            Amplitude‑encoded input vector.
        shift : float, default 0.5
            Shift magnitude for the rule.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients w.r.t. rotation_params and entangle_params
            respectively, matching their shapes.
        """
        rot = rotation_params.copy()
        ent = entangle_params.copy()
        grads_rot = np.zeros_like(rot)
        grads_ent = np.zeros_like(ent)

        # Baseline expectation (unused but shows how to compute)
        base_circ = self._build_circuit(rot, ent, inputs)
        _ = self._expectation_value(base_circ)

        # Gradient wrt rotation parameters
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                for p in range(3):
                    rot[layer, q, p] += shift
                    circ_plus = self._build_circuit(rot, ent, inputs)
                    f_plus = self._expectation_value(circ_plus)
                    rot[layer, q, p] -= 2 * shift
                    circ_minus = self._build_circuit(rot, ent, inputs)
                    f_minus = self._expectation_value(circ_minus)
                    grads_rot[layer, q, p] = (f_plus - f_minus) / (2 * shift)
                    rot[layer, q, p] += shift  # reset

        # Gradient wrt entanglement parameters
        for layer in range(self.n_layers):
            for idx in range(len(self.entangler_map)):
                ent[layer, idx] += shift
                circ_plus = self._build_circuit(rot, ent, inputs)
                f_plus = self._expectation_value(circ_plus)
                ent[layer, idx] -= 2 * shift
                circ_minus = self._build_circuit(rot, ent, inputs)
                f_minus = self._expectation_value(circ_minus)
                grads_ent[layer, idx] = (f_plus - f_minus) / (2 * shift)
                ent[layer, idx] += shift  # reset

        return grads_rot, grads_ent

__all__ = ["SelfAttentionEnhancedQ"]
