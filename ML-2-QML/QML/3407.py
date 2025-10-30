"""Quantum helper for the hybrid autoencoder.

The implementation follows a RealAmplitudes ansatz, injects a domain‑wall
pattern for feature mixing, and uses a swap‑test to compare the encoded
state with the latent vector.  The circuit is wrapped in a
:class:`qiskit_machine_learning.neural_networks.SamplerQNN` providing a
classical interface for gradients.

Key functions
-------------
- :func:`hybrid_autoencoder_qnn` – factory mirroring the classical
  ``hybrid_autoencoder`` but returning a quantum neural network.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def _domain_wall_circuit(num_qubits: int, start: int = 0) -> QuantumCircuit:
    """Inject a domain‑wall pattern on a contiguous block of qubits."""
    qc = QuantumCircuit(num_qubits)
    for i in range(start, num_qubits):
        qc.x(i)
    return qc

def hybrid_autoencoder_qnn(
    num_latent: int,
    num_trash: int,
    num_features: int,
    depth: int = 3,
    swap_test: bool = True,
    domain_wall: bool = True,
) -> SamplerQNN:
    """
    Build a quantum autoencoder circuit.

    Parameters
    ----------
    num_latent
        Number of qubits used for the latent representation.
    num_trash
        Number of auxiliary qubits for encoding and swap‑test operations.
    num_features
        Number of classical features to be encoded.
    depth
        Depth of the RealAmplitudes ansatz.
    swap_test
        If True, a swap‑test with an auxiliary qubit is appended.
    domain_wall
        If True, a domain‑wall pattern is applied before the main ansatz.
    """
    algorithm_globals.random_seed = 42

    # Feature map
    feature_params = ParameterVector("x", num_features)
    feature_circuit = RawFeatureVector(num_features)

    # Latent ansatz
    latent_params = ParameterVector("theta", num_latent * depth)
    latent_ansatz = RealAmplitudes(num_latent, reps=depth)

    # Main circuit
    total_qubits = num_latent + 2 * num_trash + 1  # +1 for swap‑test ancilla
    qc = QuantumCircuit(total_qubits)
    # Encode features
    qc.compose(feature_circuit, range(num_latent), inplace=True)
    # Domain‑wall injection
    if domain_wall:
        dw_qc = _domain_wall_circuit(total_qubits - 1, start=num_latent)
        qc.compose(dw_qc, range(num_latent, total_qubits - 1), inplace=True)
    # Latent ansatz
    qc.compose(latent_ansatz, range(num_latent), inplace=True)
    # Swap‑test with ancilla
    if swap_test:
        ancilla = total_qubits - 1
        qc.h(ancilla)
        for i in range(num_trash):
            qc.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
        qc.h(ancilla)
        qc.measure(ancilla, 0)  # measurement into classical register 0

    # Observables: expectation of Z on the ancilla gives fidelity
    observables = [SparsePauliOp("Z" + "I" * (total_qubits - 1))]

    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_params,
        weight_params=latent_params,
        interpret=lambda x: x,  # identity interpret
        output_shape=(1,),
        sampler=sampler,
    )
    return qnn

__all__ = ["hybrid_autoencoder_qnn"]
