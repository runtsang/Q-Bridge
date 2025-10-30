import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def quantum_autoencoder_qnn(num_latent: int, num_trash: int = 2, reps: int = 3) -> SamplerQNN:
    """
    Builds a swap‑test based variational circuit that estimates the fidelity
    between a latent state encoded on ``num_latent`` qubits and a reference
    |0...0⟩ state.  The latent vector is passed as weight parameters to the
    ansatz; no classical inputs are required.
    """
    # Parameter vector for the latent representation
    latent_params = ParameterVector("θ", length=num_latent)

    # Build the variational ansatz on the combined latent + trash subspace
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)

    # Construct the full circuit
    total_qubits = num_latent + num_trash + 1  # +1 ancilla
    qc = QuantumCircuit(total_qubits, 1)

    # Encode the latent vector via the ansatz
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    # Swap test with ancilla
    ancilla = num_latent + num_trash
    qc.h(ancilla)
    for i in range(num_trash):
        qc.cswap(ancilla, num_latent + i, num_latent + num_trash + i)
    qc.h(ancilla)
    qc.measure(ancilla, 0)

    # Sampler for probability estimation
    sampler = StatevectorSampler()

    # Interpret the output as fidelity: 2*P(0) - 1
    def interpret(x):
        return 2.0 * x[0] - 1.0

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=latent_params,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn

__all__ = ["quantum_autoencoder_qnn"]
