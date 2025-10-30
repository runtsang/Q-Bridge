"""Hybrid quantum autoencoder + variational classifier.

The quantum implementation mirrors the classical API.  A small
auto‑encoder built with a RealAmplitudes ansatz produces a latent
state that is fed into a depth‑controlled variational classifier.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler as Sampler

# --------------------------------------------------------------------------- #
# Quantum auto‑encoder
# --------------------------------------------------------------------------- #

def quantum_autoencoder(num_latent: int, num_trash: int) -> Tuple[QuantumCircuit, List[ParameterVector]]:
    """Return a circuit that maps input states to a compressed latent space."""
    qr = QuantumRegister(num_latent + num_trash, "q")
    qc = QuantumCircuit(qr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    return qc, ansatz.parameters

def build_autoencoder_qnn(num_latent: int, num_trash: int) -> SamplerQNN:
    """Wrap the auto‑encoder circuit in a SamplerQNN."""
    qc, weight_params = quantum_autoencoder(num_latent, num_trash)
    sampler = Sampler()
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=weight_params,
        interpret=lambda x: x,
        output_shape=(num_latent,),
        sampler=sampler,
    )

# --------------------------------------------------------------------------- #
# Variational classifier
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Return a layered ansatz that acts on the latent qubits."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class HybridAutoencoderClassifier:
    """Quantum pipeline: auto‑encoder → latent state → variational classifier."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int = 2,
        clf_depth: int = 3,
    ) -> None:
        self.autoencoder = build_autoencoder_qnn(num_latent, num_trash)
        self.classifier_circuit, _, _, _ = build_classifier_circuit(num_latent, clf_depth)

    def __call__(self, input_state: Statevector) -> List[float]:
        """Apply the auto‑encoder, then the classifier ansatz, and return the latent amplitudes."""
        latent = self.autoencoder(input_state)
        # For demonstration, we simply return the latent amplitudes.
        return [float(v.real) for v in latent]

__all__ = [
    "quantum_autoencoder",
    "build_autoencoder_qnn",
    "build_classifier_circuit",
    "HybridAutoencoderClassifier",
]
