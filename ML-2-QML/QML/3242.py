"""Quantum component of the hybrid autoencoder.

The module exposes:
* :func:`quantum_latent_circuit` – a RealAmplitudes ansatz.
* :class:`QuantumLatentLayer` – a qiskit SamplerQNN that returns expectation values.
* :func:`FCL` – a lightweight quantum fully‑connected layer that can be plugged
  into the classical decoder for an all‑quantum decoder head.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def quantum_latent_circuit(num_qubits: int, reps: int) -> QuantumCircuit:
    """
    Build a parameterized ansatz that will generate the latent vector.
    Each qubit will be measured in the Z basis to produce an expectation value.
    """
    qc = QuantumCircuit(num_qubits)
    qc.compose(RealAmplitudes(num_qubits, reps=reps), inplace=True)
    return qc


class QuantumLatentLayer:
    """
    Wraps a :class:`SamplerQNN` around the latent circuit.
    The layer is fully differentiable with respect to its parameters.
    """
    def __init__(self, num_qubits: int, reps: int, shots: int = 1024) -> None:
        self.circuit = quantum_latent_circuit(num_qubits, reps)
        self.sampler = Sampler(Aer.get_backend("aer_simulator"))
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=(num_qubits,),
            sampler=self.sampler,
        )

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """
        Forward pass through the SamplerQNN.
        params: shape (batch, num_qubits)
        Returns: shape (batch, num_qubits)
        """
        return self.qnn(params)


def FCL(num_qubits: int = 1, shots: int = 1000) -> QuantumCircuit:
    """
    Simple quantum circuit that mimics a fully‑connected layer.
    The circuit prepares a uniform superposition, applies a parametrized
    rotation, and measures the expectation value of Z.
    """
    qc = QuantumCircuit(num_qubits)
    theta = qiskit.circuit.Parameter("theta")
    qc.h(range(num_qubits))
    qc.ry(theta, range(num_qubits))
    qc.measure_all()
    return qc
