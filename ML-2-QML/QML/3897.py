from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


def create_sampler_qnn(
    num_qubits: int,
    reps: int = 5,
    entanglement: str = "circular",
    output_shape: int = 1,
) -> SamplerQNN:
    """Return a SamplerQNN wrapping a RealAmplitudes ansatz.

    The ansatz is parameterised by ``num_qubits`` and ``reps`` and uses the
    specified entanglement pattern. The QNN exposes a single scalar output
    that is interpreted as the expectation value of the first basis state.
    """
    ansatz = RealAmplitudes(
        num_qubits,
        reps=reps,
        entanglement=entanglement,
    )
    qnn = SamplerQNN(
        circuit=ansatz,
        input_params=[],          # latent vector is fed externally
        weight_params=ansatz.parameters,
        interpret=lambda x: x[0],  # scalar output
        output_shape=output_shape,
        sampler=StatevectorSampler(),
    )
    return qnn


__all__ = ["create_sampler_qnn"]
