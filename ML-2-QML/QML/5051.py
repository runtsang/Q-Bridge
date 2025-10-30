"""
Quantum autoencoder component for the hybrid model.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit import Aer
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# Quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def _quantum_autoencoder(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Variational encoder–decoder circuit with a swap‑test style disentanglement."""
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode
    qc.append(RealAmplitudes(num_qubits, reps=reps), range(num_qubits))
    qc.barrier()

    # Auxiliary qubit for swap test
    aux = num_qubits
    qc.add_register(QuantumRegister(1, "aux"))
    qc.h(aux)

    # Swap test between each data qubit and an auxiliary “trash” qubit
    for i in range(num_qubits):
        qc.cswap(aux, i, i)

    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------------- #
# QNN wrapper
# --------------------------------------------------------------------------- #
def AutoencoderGen380QNN(num_qubits: int, latent_dim: int) -> SamplerQNN:
    """
    Build a QNN that acts as a quantum encoder producing a *latent_dim*‑dimensional
    feature vector. The output is interpreted as a real‑valued vector using
    a simple linear mapping on the measurement probabilities.
    """
    # Circuits
    encoder_circuit = _quantum_autoencoder(num_qubits, reps=4)
    # Parameters are the rotation angles in the RealAmplitudes ansatz
    weight_params = encoder_circuit.parameters

    # Simple linear interpret: map probability of |1⟩ on ancilla to latent vector
    def interpret(x: np.ndarray) -> np.ndarray:
        # x is a 1‑dim array of counts for 0/1 measurement of ancilla
        probs = x / x.sum()
        return np.array([probs[1]] * latent_dim, dtype=np.float32)

    # Sampler
    sampler = StatevectorSampler(Aer.get_backend("aer_simulator_statevector"))

    # Build QNN
    qnn = SamplerQNN(
        circuit=encoder_circuit,
        input_params=[],          # No classical input – circuit encodes data via state preparation
        weight_params=weight_params,
        interpret=interpret,
        output_shape=(latent_dim,),
        sampler=sampler,
    )
    return qnn

__all__ = ["AutoencoderGen380QNN", "_quantum_autoencoder"]
