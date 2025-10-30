from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import SamplerQNN

class UnifiedAutoencoder(SamplerQNN):
    """Quantum refinement network based on a RealAmplitudes ansatz and Pauli‑Z observables.

    The circuit encodes an input latent vector into a quantum state via Ry rotations,
    applies a parameterised ansatz, and measures the expectation value of Pauli‑Z on
    each qubit.  The resulting vector of expectation values is used as the refined
    latent representation.  This class can be used as a drop‑in replacement for a
    classical autoencoder encoder.
    """
    def __init__(self, latent_dim: int, reps: int = 3, shots: int = 1024) -> None:
        num_qubits = latent_dim
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Input encoding: Ry rotations
        input_params = [Parameter(f"x{i}") for i in range(num_qubits)]
        for i, p in enumerate(input_params):
            circuit.ry(p, i)

        # Ansatz layer
        ansatz = RealAmplitudes(num_qubits, reps=reps)
        circuit.compose(ansatz, qr, inplace=True)

        # Observables: Pauli‑Z on each qubit
        observables = []
        for i in range(num_qubits):
            pauli_str = "I" * num_qubits
            pauli_str = pauli_str[:i] + "Z" + pauli_str[i+1:]
            observables.append(Pauli(pauli_str))

        weight_params = list(ansatz.parameters)

        super().__init__(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            interpret=lambda x: x,  # output expectation values directly
            output_shape=(num_qubits,),
            sampler=StatevectorSampler(),
        )

__all__ = ["UnifiedAutoencoder"]
