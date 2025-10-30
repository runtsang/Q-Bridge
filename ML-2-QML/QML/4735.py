from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector

def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Insert an X‑domain wall over qubits [start, end)."""
    for i in range(start, end):
        circuit.x(i)
    return circuit

def HybridSamplerQNN(num_latent: int = 3,
                     num_trash: int = 2,
                     seed: int = 42) -> SamplerQNN:
    """
    Quantum sampler that mirrors the classical HybridSamplerQNN:
    * A RealAmplitudes ansatz encodes latent features.
    * A swap test with auxiliary qubits implements a trash‑qubit strategy.
    * A domain‑wall block injects structured noise.
    The circuit returns a two‑outcome probability vector via a state‑vector sampler.
    """
    # Quantum registers
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode latent part with a variational ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test trash‑qubit scheme
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    # Inject a deterministic domain wall
    circuit = _domain_wall(circuit, 0, len(qr))

    # Build the SamplerQNN wrapper
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,  # identity: raw sample probabilities
        output_shape=2,
        sampler=sampler
    )
    return qnn

__all__ = ["HybridSamplerQNN"]
