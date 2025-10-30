"""Quantum classifier that mirrors the classical surrogate."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# Import quantum sub‑modules from the seed files
from QuantumClassifierModel import build_classifier_circuit
from SelfAttention import SelfAttention
from SamplerQNN import SamplerQNN


class QuantumClassifierModel:
    """
    Quantum implementation of the hybrid classifier.

    The circuit is composed of:
        - Data encoding via RX rotations.
        - A quantum self‑attention sub‑circuit.
        - A sampler circuit that produces a 2‑qubit output.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        kernel_size: int = 2,
        embed_dim: int = 4,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim

        # Parameter vectors
        self.data_params = ParameterVector("x", num_qubits)
        self.rot_params = ParameterVector("r", embed_dim * embed_dim)
        self.ent_params = ParameterVector("e", embed_dim * embed_dim)
        self.weight_params = ParameterVector("w", 4)

        # Build the full circuit
        self.circuit, self.obs = self._build_full_circuit()

        # Backend
        self.backend: Backend = AerSimulator()

    def _build_full_circuit(self) -> tuple[QuantumCircuit, list[SparsePauliOp]]:
        """
        Construct the complete variational circuit with self‑attention and sampler.
        """
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(self.num_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # 1. Data encoding
        for i in range(self.num_qubits):
            circuit.rx(self.data_params[i], qr[i])

        # 2. Self‑attention block
        attn = SelfAttention(self.num_qubits)
        attn_circuit = attn._build_circuit(self.rot_params, self.ent_params)
        circuit.append(attn_circuit, qr)

        # 3. Sampler sub‑circuit (2‑qubit output)
        sampler_circuit = SamplerQNN()
        # sampler_circuit is a Qiskit circuit, we add it directly
        circuit.append(sampler_circuit, qr)

        # 4. Measurement
        circuit.measure(qr, cr)

        # Observables for classification
        observables = [
            SparsePauliOp("Z" + "I" * (self.num_qubits - 1)),
            SparsePauliOp("I" * (self.num_qubits - 1) + "Z"),
        ]

        return circuit, observables

    def run(
        self,
        data: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit on a batch of data.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (batch, num_qubits).
        shots : int
            Number of shots per execution.

        Returns
        -------
        np.ndarray
            Class probabilities of shape (batch, 2).
        """
        batch_size = data.shape[0]
        probs = np.zeros((batch_size, 2))

        for idx in range(batch_size):
            # Bind data parameters
            bind_dict = {self.data_params[i]: data[idx, i] for i in range(self.num_qubits)}
            bound_circuit = self.circuit.bind_parameters(bind_dict)

            # Execute
            job = execute(bound_circuit, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts(bound_circuit)

            # Convert counts to probabilities of measuring |1> on each of the last two qubits
            for key, val in counts.items():
                # key is bitstring; last two bits correspond to output qubits
                prob1 = int(key[-2]) * val / shots
                prob2 = int(key[-1]) * val / shots
                probs[idx, 0] += prob1
                probs[idx, 1] += prob2

        # Normalise
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


__all__ = ["QuantumClassifierModel"]
