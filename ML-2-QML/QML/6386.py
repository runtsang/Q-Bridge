import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42


def _real_amplitude_ansatz(num_qubits: int, reps: int = 5) -> QuantumCircuit:
    """Parameterized ansatz for the encoder/decoder part."""
    return RealAmplitudes(num_qubits, reps=reps)


def _autoencoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """Builds a circuit that implements a quantum auto‑encoder with a swap‑test
    measuring overlap between the latent and a reference state."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode latent + trash
    circuit.compose(_real_amplitude_ansatz(num_latent + num_trash, reps),
                    range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test with auxiliary qubit
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


def _domain_wall_circuit(num_qubits: int, start: int, end: int) -> QuantumCircuit:
    """Injects a domain‑wall pattern by applying X gates to a contiguous block."""
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc


class HybridAutoencoder:
    """Quantum neural network that mirrors the hybrid classical architecture.

    Depending on ``use_estimator`` the object contains either a
    :class:`SamplerQNN` representing the full auto‑encoder circuit or an
    :class:`EstimatorQNN` that performs a simple regression on the
    auxiliary qubit.  The returned QNN can be used directly in a
    quantum‑classical training loop.
    """

    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        use_estimator: bool = True,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.use_estimator = use_estimator
        self.qnn = self._build_qnn()

    def _build_qnn(self):
        ae_circuit = _autoencoder_circuit(
            self.num_latent, self.num_trash, self.reps
        )
        dw_circuit = _domain_wall_circuit(
            self.num_latent + 2 * self.num_trash,
            self.num_latent,
            self.num_latent + self.num_trash,
        )
        ae_circuit.compose(dw_circuit, range(self.num_latent + self.num_trash),
                           inplace=True)

        if self.use_estimator:
            param = Parameter("theta")
            est_circuit = QuantumCircuit(1, 1)
            est_circuit.ry(param, 0)
            est_circuit.measure(0, 0)
            observable = SparsePauliOp.from_list([("Y" * est_circuit.num_qubits, 1)])
            estimator = StatevectorEstimator()
            return QEstimatorQNN(
                circuit=est_circuit,
                observables=observable,
                input_params=[],
                weight_params=[param],
                estimator=estimator,
            )
        else:
            sampler = StatevectorSampler()
            return SamplerQNN(
                circuit=ae_circuit,
                input_params=[],
                weight_params=ae_circuit.parameters,
                interpret=lambda x: x,
                output_shape=2,
                sampler=sampler,
            )

    def get_qnn(self):
        """Return the underlying quantum neural network."""
        return self.qnn


__all__ = ["HybridAutoencoder"]
