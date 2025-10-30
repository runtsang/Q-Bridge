import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit import ClassicalRegister, QuantumRegister

class QuantumConv:
    """Variational quanvolution circuit that encodes a 2‑D patch."""
    def __init__(self, kernel_size: int = 3, shots: int = 1024,
                 threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.theta = ParameterVector('θ', self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, t in enumerate(self.theta):
            self.circuit.rx(t, i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0
                    for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = execute(self.circuit, self.backend,
                      parameter_binds=param_binds,
                      shots=self.shots)
        result = job.result().get_counts(self.circuit)
        counts = sum([sum(int(b) for b in key) * val for key, val in result.items()])
        return counts / (self.shots * self.n_qubits)

class QuantumAutoencoder:
    """Quantum autoencoder built with a sampler neural network."""
    def __init__(self, latent: int = 3, trash: int = 2) -> None:
        self.latent = latent
        self.trash = trash
        self.qc = self._make_circuit()

    def _make_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent + 2 * self.trash + 1, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)
        # ansatz
        qc.compose(RealAmplitudes(self.latent + self.trash), range(self.latent + self.trash), inplace=True)
        qc.barrier()
        aux = self.latent + 2 * self.trash
        qc.h(aux)
        for i in range(self.trash):
            qc.cswap(aux, self.latent + i, self.latent + self.trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def sample(self, data: np.ndarray) -> np.ndarray:
        # For simplicity, use the circuit as a sampler to output a single expectation
        job = execute(self.qc, Aer.get_backend('qasm_simulator'),
                      shots=1024, params={})
        result = job.result().get_counts(self.qc)
        mean = sum([int(bit) * count for bit, count in result.items()]) / 1024
        return np.array([mean])

class QuantumKernel:
    """TorchQuantum‑style kernel that evaluates overlap of encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.qc = self._make_ansatz()

    def _make_ansatz(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_wires, 'q')
        qc = QuantumCircuit(qr)
        for i in range(self.n_wires):
            qc.ry(i, qr[i])
        return qc

    def run(self, x: np.ndarray, y: np.ndarray) -> float:
        # Encode x, then reverse encode y, measure overlap
        qc = self.qc.copy()
        # encode x
        for i, val in enumerate(x):
            qc.ry(val, i)
        # decode y (reverse parameters)
        for i, val in enumerate(y):
            qc.ry(-val, i)
        job = execute(qc, Aer.get_backend('statevector_simulator'))
        sv = job.result().get_statevector(qc)
        return np.abs(sv[0]) ** 2

class SharedClassName:
    """Quantum counterpart of the hybrid pipeline."""
    def __init__(self, conv_k: int = 3, latent: int = 3,
                 trash: int = 2, kernel_wires: int = 4) -> None:
        self.conv = QuantumConv(kernel_size=conv_k)
        self.auto = QuantumAutoencoder(latent=latent, trash=trash)
        self.kernel = QuantumKernel(n_wires=kernel_wires)

    def forward(self, data: np.ndarray):
        """
        Parameters
        ----------
        data : np.ndarray
            Input image of shape (H, W).

        Returns
        -------
        recon : np.ndarray
            Reconstructed image (quantum autoencoder output).
        sims : np.ndarray
            Similarity score from the quantum kernel.
        """
        conv_out = self.conv.run(data)                     # scalar
        z = self.auto.sample(np.array([conv_out]))          # latent sample
        recon = self.auto.sample(z)                        # reconstruct from latent
        sims = self.kernel.run(z, z)                       # kernel similarity
        return recon, sims

__all__ = ["SharedClassName"]
