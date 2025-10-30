from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

def quantum_decoder_circuit(latent_dim: int, output_dim: int) -> QuantumCircuit:
    """
    Construct a variational circuit that takes a latent vector as input
    parameters and produces a reconstruction through expectation values.
    """
    circuit = QuantumCircuit(latent_dim + output_dim)

    # Input parameters from latent vector
    latent_params = [Parameter(f"latent_{i}") for i in range(latent_dim)]
    for i, p in enumerate(latent_params):
        circuit.ry(p, i)

    # Variational ansatz on output qubits
    ansatz = RealAmplitudes(output_dim, reps=3)
    circuit.append(ansatz, range(latent_dim, latent_dim + output_dim))
    weight_params = list(ansatz.parameters)

    # Observables: Pauli X on each output qubit
    observables = [
        SparsePauliOp.from_list([(f"X{'I' * (output_dim - 1 - i)}", 1)])
        for i in range(output_dim)
    ]

    estimator = Estimator()
    return EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=latent_params,
        weight_params=weight_params,
        estimator=estimator,
    )

def QuantumDecoderQNN(latent_dim: int, output_dim: int) -> EstimatorQNN:
    """Convenience wrapper to expose the quantum decoder."""
    return quantum_decoder_circuit(latent_dim, output_dim)

__all__ = ["QuantumDecoderQNN", "quantum_decoder_circuit"]
