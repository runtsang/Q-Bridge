import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import ParameterVector

def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class AutoencoderHybrid:
    """Quantum‑centric hybrid autoencoder."""
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 depth: int = 2,
                 num_qubits: int = 4,
                 backend: str = "statevector_simulator"):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.depth = depth
        self.num_qubits = num_qubits
        self.backend = backend
        algorithm_globals.random_seed = 42
        self.sampler = StatevectorSampler()
        self.estimator = StatevectorEstimator()

    def build_encoder_circuit(self) -> QuantumCircuit:
        """Encode classical data into a quantum state and perform a swap test to extract latent."""
        qr = QuantumRegister(self.num_latent + self.num_trash, "q")
        cr = ClassicalRegister(self.num_latent, "c")
        circuit = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(self.num_latent + self.num_trash), inplace=True)
        aux = self.num_latent + self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def build_decoder_circuit(self) -> QuantumCircuit:
        """Decode a latent vector back into the original dimension using a variational ansatz."""
        qr = QuantumRegister(self.num_latent + self.num_trash + 1, "q")
        cr = ClassicalRegister(self.num_latent, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(0)
        for i in range(1, self.num_latent + 1):
            circuit.cx(0, i)
        ansatz = RealAmplitudes(self.num_trash, reps=3)
        circuit.compose(ansatz, range(self.num_latent + 1, self.num_latent + self.num_trash + 1), inplace=True)
        aux = self.num_latent + self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, i, self.num_latent + 1 + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def domain_wall(self, circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        """Apply a domain wall (X gates) between qubits a and b."""
        for i in range(a, b):
            circuit.x(i)
        return circuit

    def build_classifier_circuit(self):
        """Return a quantum classifier circuit mirroring the classical build_classifier_circuit."""
        return build_classifier_circuit(self.num_qubits, self.depth)

    def get_sampler_qnn(self):
        """Wrap the decoder circuit in a SamplerQNN."""
        circuit = self.build_decoder_circuit()
        return QSamplerQNN(circuit=circuit,
                           input_params=[],
                           weight_params=circuit.parameters,
                           sampler=self.sampler,
                           output_shape=2,
                           interpret=lambda x: x)

    def get_estimator_qnn(self):
        """Wrap the encoder circuit in an EstimatorQNN."""
        circuit = self.build_encoder_circuit()
        observables = [SparsePauliOp.from_list([("Z", 1)])]
        return QEstimatorQNN(circuit=circuit,
                              observables=observables,
                              input_params=[],
                              weight_params=circuit.parameters,
                              estimator=self.estimator)

    def run(self, data: np.ndarray):
        """Placeholder run: encode and decode using the quantum circuits.
        For a full implementation, parameters would be set from `data` and the circuits
        executed on the chosen backend.  Here we simply return a zero array of appropriate shape."""
        return np.zeros((len(data), self.num_latent))

__all__ = ["AutoencoderHybrid", "build_classifier_circuit"]
