from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit import algorithm_globals

def HybridEstimatorQNN_QML(
    num_latent: int = 3,
    num_trash: int = 2,
    reps: int = 3,
) -> EstimatorQNN:
    """Quantum hybrid estimator that embeds a variational ansatz and a swap‑test."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        """Variational circuit for encoding the classical input."""
        return RealAmplitudes(num_qubits, reps=reps)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Auto‑encoder style circuit with swap‑test."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode the input on the first num_latent qubits
        qc.compose(ansatz(num_latent), range(0, num_latent), inplace=True)

        # Swap‑test with the trash qubits
        qc.barrier()
        auxiliary = num_latent + 2 * num_trash
        qc.h(auxiliary)
        for i in range(num_trash):
            qc.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        qc.h(auxiliary)
        qc.measure(auxiliary, cr[0])

        return qc

    circuit = auto_encoder_circuit(num_latent, num_trash)
    # Input parameters correspond to the variational parameters of the ansatz
    input_params = [Parameter(f"θ{i}") for i in range(circuit.num_parameters)]
    # All other parameters are treated as weights for the swap‑test
    weight_params = circuit.parameters

    # Observable: expectation of the Y operator on the auxiliary qubit
    observable = SparsePauliOp.from_list([("Y" * circuit.num_qubits, 1)])

    estimator = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=StatevectorEstimator(),
    )
    return estimator

__all__ = ["HybridEstimatorQNN_QML"]
