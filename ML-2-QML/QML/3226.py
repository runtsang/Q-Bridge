"""Hybrid quantum autoencoder with variational circuit and quantum convolutional filter."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

# ----------------------------------------------------------------------
# Quantum convolutional filter
# ----------------------------------------------------------------------
class QuanvCircuit:
    """Quantum filter emulating a 2x2 convolution with a random circuit."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Run the filter on 2D data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        counts = job.result().get_counts(self._circuit)
        prob = 0.0
        for key, val in counts.items():
            ones = sum(int(b) for b in key)
            prob += ones * val
        return prob / (self.shots * self.n_qubits)

# ----------------------------------------------------------------------
# Quantum autoencoder circuit
# ----------------------------------------------------------------------
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Constructs a swap-test based quantum autoencoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# ----------------------------------------------------------------------
# Quantum autoencoder QNN
# ----------------------------------------------------------------------
def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2, shots: int = 1024):
    """Builds a SamplerQNN that implements a quantum autoencoder."""
    sampler = Sampler()
    circuit = autoencoder_circuit(num_latent, num_trash)
    def interpret(x):
        # Interpret measurement result as probability of |0>
        return 1.0 - x[0]
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn

__all__ = ["QuanvCircuit", "autoencoder_circuit", "QuantumAutoencoder"]
