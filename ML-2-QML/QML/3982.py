import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class ConvGen302:
    """Quantum implementation of a hybrid convolution–autoencoder filter."""
    def __init__(
        self,
        kernel_size: int = 2,
        latent_dim: int = 3,
        trash: int = 2,
        shots: int = 100,
        threshold: float = 0.5,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.trash = trash
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Variational ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.trash, reps=5)
        circ.compose(ansatz, range(0, self.latent_dim + self.trash), inplace=True)

        circ.barrier()

        aux = self.latent_dim + 2 * self.trash
        circ.h(aux)
        for i in range(self.trash):
            circ.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])

        return circ

    def run(self, data: np.ndarray) -> float:
        n_qubits = self.latent_dim + 2 * self.trash
        param_binds = []
        for val in data.reshape(-1, n_qubits):
            bind = {}
            for i, v in enumerate(val):
                bind[Parameter(f"theta{i}")] = np.pi if v > self.threshold else 0.0
            param_binds.append(bind)
        job = execute(self.circuit, backend=self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        total = sum(cnt * sum(int(b) for b in bitstring) for bitstring, cnt in counts.items())
        return total / (self.shots * n_qubits * len(param_binds))

    def autoencoder_qnn(self) -> SamplerQNN:
        sampler = Sampler()
        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )
        return qnn
