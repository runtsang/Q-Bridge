"""Quantum hybrid attention sampler using Qiskit and Pennylane.

The module constructs a parameterized circuit that first applies
rotation gates (mimicking the attention query/key/value transformations),
then entangles qubits to capture dependencies, and finally performs a
sampling routine that outputs a probability distribution.  The circuit
is compatible with both the Aer simulator and real backends.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as QiskitSampler


class QuantumHybridAttentionSampler:
    """Parameterised circuit that fuses attention‑style rotations with a sampler."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        sampler_params: np.ndarray,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)

        # Attention‑style rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entanglement to capture dependencies
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)

        # Sampler section (parameterised Ry gates)
        for i in range(self.n_qubits):
            qc.ry(sampler_params[i], i)

        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        sampler_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the hybrid circuit and return measurement counts.

        Parameters
        ----------
        rotation_params, entangle_params, sampler_params : np.ndarray
            Parameter arrays for the circuit.
        shots : int, default 1024
            Number of shots for sampling.

        Returns
        -------
        dict
            Measurement outcome counts.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, sampler_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

    def sampler_qnn(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        sampler_params: np.ndarray,
    ) -> SamplerQNN:
        """
        Wrap the circuit in a Qiskit Machine Learning SamplerQNN for
        integration with gradient‑based optimisers.

        Returns
        -------
        SamplerQNN
            A Qiskit SamplerQNN instance ready for training.
        """
        input_params = ParameterVector("input", self.n_qubits)
        weight_params = ParameterVector("weight", self.n_qubits)

        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        for i in range(self.n_qubits):
            qc.ry(sampler_params[i], i)

        sampler = QiskitSampler()
        return SamplerQNN(circuit=qc, input_params=input_params, weight_params=weight_params, sampler=sampler)


__all__ = ["QuantumHybridAttentionSampler"]
