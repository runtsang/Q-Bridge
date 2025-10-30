"""Quantum‑enhanced self‑attention helper.

The module defines a SelfAttentionGen010Q class that mirrors the
classical `SelfAttentionGen010` API but implements the attention
weights via a variational `SamplerQNN` circuit.  The circuit uses a
RealAmplitudes ansatz for rotations and a controlled‑swap (CSWAP)
pattern for entanglement, inspired by the Autoencoder and
SamplerQNN seed examples.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RealAmplitudes, LinearEntangler
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class SelfAttentionGen010Q:
    """
    Quantum self‑attention layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits that encode the feature dimension.
    reps : int, default 3
        Number of repetitions for the RealAmplitudes ansatz.
    """

    def __init__(self, n_qubits: int, *, reps: int = 3) -> None:
        self.n_qubits = n_qubits
        self.reps = reps
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)
        # Build the parameterised ansatz
        self.ansatz = RealAmplitudes(n_qubits, reps=reps)
        self.circuit.compose(self.ansatz, inplace=True)
        # Entanglement via CSWAP pattern
        for i in range(n_qubits - 1):
            self.circuit.cswap(i, i + 1, i)
        self.circuit.measure(self.qr, self.cr)

        # SamplerQNN instance
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.ansatz.parameters,
            sampler=self.sampler
        )

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Inject parameters into the ansatz and entanglement pattern.
        """
        circuit = self.circuit.copy()
        # Map rotation parameters onto the ansatz
        param_dict = {p: v for p, v in zip(self.ansatz.parameters, rotation_params)}
        circuit.assign_parameters(param_dict, inplace=True)
        # Entanglement parameters are applied as controlled‑rx gates
        for i, val in enumerate(entangle_params):
            if i < self.n_qubits - 1:
                circuit.cx(i, i + 1)
                circuit.rx(val, i + 1)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the quantum attention circuit and return the sampled
        attention distribution.

        Parameters
        ----------
        backend : qiskit.providers.BaseBackend
            Backend on which to execute the circuit.
        rotation_params : np.ndarray
            Rotation angles for the ansatz.
        entangle_params : np.ndarray
            Parameters for the entanglement pattern.
        inputs : np.ndarray
            Input features (unused in this simple example but kept for API).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Attention weight vector of shape (n_qubits,).
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend=backend, shots=shots)
        counts = job.result().get_counts(circuit)
        probs = np.zeros(self.n_qubits)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            probs[idx] = count
        probs /= probs.sum()
        return probs

    def __repr__(self) -> str:
        return f"SelfAttentionGen010Q(n_qubits={self.n_qubits}, reps={self.reps})"

__all__ = ["SelfAttentionGen010Q"]
