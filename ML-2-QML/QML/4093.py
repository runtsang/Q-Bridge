"""Quantum hybrid self‑attention using a parameterized sampler QNN and a quantum‑kernel inspired encoder."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class HybridSelfAttention:
    """
    Quantum self‑attention that encodes input vectors via a quantum‑kernel
    ansatz and then applies a sampler QNN to produce a discrete attention
    distribution. The circuit is fully parameterized and can be executed
    on any Qiskit backend.
    """

    def __init__(self, n_qubits: int, embed_dim: int) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        embed_dim : int
            Dimensionality of the classical input vectors.
        """
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.backend = Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler()

    def _create_sampler_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        input_vector: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a parameterized circuit that encodes the input vector,
        applies rotation and entanglement layers, and ends with a
        sampler QNN.
        """
        # Parameter vectors for the sampler
        input_params = ParameterVector("input", self.embed_dim)
        weight_params = ParameterVector("weight", 4)

        qc = QuantumCircuit(self.n_qubits)

        # Encode the classical input using Ry rotations
        for i, val in enumerate(input_vector):
            qc.ry(val, i % self.n_qubits)

        # Rotation layer driven by rotation_params
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer driven by entangle_params
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)

        # Sampler QNN block
        qc.ry(weight_params[0], 0)
        qc.ry(weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(weight_params[2], 0)
        qc.ry(weight_params[3], 1)
        qc.cx(0, 1)

        qc.measure_all()
        return qc, input_params, weight_params

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the hybrid quantum self‑attention over a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of input vectors, shape (batch, embed_dim).
        rotation_params : np.ndarray
            Rotation parameters for the circuit, shape (n_qubits, 3).
        entangle_params : np.ndarray
            Entanglement parameters, shape (n_qubits - 1,).
        shots : int, optional
            Number of shots per circuit.  Defaults to 1024.

        Returns
        -------
        np.ndarray
            Attention weight distribution for each input, shape (batch, 2**n_qubits).
        """
        batch_size = inputs.shape[0]
        distributions = []

        for idx in range(batch_size):
            qc, input_params, weight_params = self._create_sampler_circuit(
                rotation_params, entangle_params, inputs[idx]
            )
            sampler_qnn = SamplerQNN(
                circuit=qc,
                input_params=input_params,
                weight_params=weight_params,
                sampler=self.sampler,
            )
            # Sample once per input; weights are set to zero for simplicity
            samples = sampler_qnn.sample(
                inputs=inputs[idx].reshape(1, -1),
                weights=np.zeros((1, 4)),
                shots=shots,
            )
            # Convert samples to a probability distribution
            probs = np.zeros(2 ** self.n_qubits)
            for sample in samples:
                bitstring = "".join(str(b) for b in sample)
                probs[int(bitstring, 2)] += 1
            probs /= shots
            distributions.append(probs)

        return np.array(distributions)
