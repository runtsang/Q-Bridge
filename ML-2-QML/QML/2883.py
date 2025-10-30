import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.optimizers import COBYLA

class ConvFilter:
    """Quantum convolutional filter that emulates a quanvolution layer."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 shots: int = 100) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Parameterised circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """
        Execute the filter on a 2‑D patch.

        Returns the average probability of measuring |1> across all qubits.
        """
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0
                        for i, val in enumerate(dat)}
                       for dat in flat]
        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)


class QuantumAutoencoder:
    """
    A toy quantum autoencoder that employs a swap‑test style circuit
    with a RealAmplitudes ansatz and a domain‑wall domain.
    """
    def __init__(self,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 reps: int = 5,
                 shots: int = 1024) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.sampler = Sampler(self.backend)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz over latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.compose(ansatz, list(range(self.num_latent + self.num_trash)), inplace=True)
        qc.barrier()

        # Swap‑test with an auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def run(self, data: np.ndarray | list[float]) -> float:
        """
        Evaluate the circuit for a single latent vector.

        Returns the probability of measuring |1> on the auxiliary qubit.
        """
        param_binds = [{p: v for p, v in zip(self.circuit.parameters, data)}]
        result: SamplerResult = self.sampler.run(self.circuit,
                                                 param_binds,
                                                 shots=self.shots)
        counts = result.get_counts()
        p1 = sum(cnt for bit, cnt in counts.items() if bit == "1")
        return p1 / self.shots

    def encode(self, data: np.ndarray | list[float]) -> float:
        return self.run(data)

    def decode(self, z: float) -> float:
        # In this toy example the decoder is identity.
        return z


class HybridConvAutoencoder:
    """
    Quantum‑only counterpart of the classical HybridConvAutoencoder.
    Both expose the same public API: encode, decode, forward and train_autoencoder.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.5,
                 conv_shots: int = 100,
                 ae_latent: int = 3,
                 ae_trash: int = 2,
                 ae_reps: int = 5,
                 ae_shots: int = 1024) -> None:
        self.conv = ConvFilter(kernel_size=conv_kernel,
                               threshold=conv_threshold,
                               shots=conv_shots)
        self.autoencoder = QuantumAutoencoder(num_latent=ae_latent,
                                              num_trash=ae_trash,
                                              reps=ae_reps,
                                              shots=ae_shots)

    def encode(self, data: np.ndarray | list[list[float]]) -> float:
        conv_out = self.conv.run(data)
        # Treat the single scalar as a 1‑D latent vector
        return self.autoencoder.encode([conv_out])

    def decode(self, z: float) -> float:
        return self.autoencoder.decode(z)

    def forward(self, data: np.ndarray | list[list[float]]) -> float:
        return self.decode(self.encode(data))

    def train_autoencoder(self,
                          data: list[float],
                          epochs: int = 50,
                          lr: float = 0.01) -> np.ndarray:
        """
        Gradient‑free optimisation of the autoencoder weights using COBYLA.
        The objective is mean‑squared error between the circuit output and the target data.
        """
        optimizer = COBYLA(maxiter=epochs)
        params = np.random.rand(self.autoencoder.circuit.num_parameters)

        def objective(p: np.ndarray) -> float:
            self.autoencoder.circuit.assign_parameters(p, inplace=True)
            preds = [self.autoencoder.run(d) for d in data]
            loss = sum((pred - d) ** 2 for pred, d in zip(preds, data))
            return loss

        params_opt = optimizer.optimize(len(params), objective, initial_point=params)
        self.autoencoder.circuit.assign_parameters(params_opt, inplace=True)
        return params_opt


__all__ = ["ConvFilter", "QuantumAutoencoder", "HybridConvAutoencoder"]
