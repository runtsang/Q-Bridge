"""Quantum autoencoder sampler implementation."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

def _autoencoder_circuit(num_latent: int, num_trash: int) -> tuple[QuantumCircuit, ParameterVector]:
    """
    Builds a quantum circuit that encodes a latent vector, entangles it with
    auxiliary qubits, and performs a swap‑test style measurement to recover
    the latent state.  The circuit is intentionally simple yet expressive
    enough to serve as a quantum autoencoder.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the latent vector into the first `num_latent` qubits
    latent_params = ParameterVector("latent", num_latent)
    qc.compose(RealAmplitudes(num_latent, reps=3), range(num_latent), inplace=True)

    # Prepare auxiliary (trash) qubits in a Bell‑like state
    for i in range(num_trash):
        qc.h(num_latent + i)
        qc.h(num_latent + num_trash + i)

    # Swap test with an ancilla
    ancilla = num_latent + 2 * num_trash
    qc.h(ancilla)
    for i in range(num_trash):
        qc.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
    qc.h(ancilla)
    qc.measure(ancilla, cr[0])

    return qc, latent_params

def QuantumSamplerQNN(
    num_latent: int = 3,
    num_trash: int = 2,
) -> SamplerQNN:
    """
    Constructs and returns a SamplerQNN that samples from the quantum
    autoencoder circuit defined above.  The returned object can be used
    directly in a hybrid training loop.
    """
    qc, latent_params = _autoencoder_circuit(num_latent, num_trash)
    sampler = Sampler()
    weight_params = [p for p in qc.parameters if p not in latent_params]
    return SamplerQNN(
        circuit=qc,
        input_params=latent_params,
        weight_params=weight_params,
        sampler=sampler,
        interpret=lambda x: x,
        output_shape=(2,),
    )

__all__ = ["QuantumSamplerQNN"]
