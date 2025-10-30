import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderGen470QNN:
    """Quantum autoencoder using a RealAmplitudes ansatz and a swap‑test decoder."""
    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, reps: int = 5):
        algorithm_globals.random_seed = 42
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(2,),
            sampler=self.sampler,
        )

    def _ansatz(self, num_qubits: int):
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _auto_encoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def _domain_wall(self, circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        for i in range(int(b / 2), int(b)):
            circuit.x(i)
        return circuit

    def _build_circuit(self) -> QuantumCircuit:
        ae = self._auto_encoder_circuit(self.latent_dim, self.trash_dim)
        dw = self._domain_wall(QuantumCircuit(self.latent_dim + 2 * self.trash_dim), 0, self.latent_dim + 2 * self.trash_dim)
        qc = QuantumCircuit(self.latent_dim + 2 * self.trash_dim + 1, 1)
        qc.compose(dw, range(self.latent_dim + self.trash_dim), inplace=True)
        qc.compose(ae, inplace=True)
        return qc

    def fidelity_loss(self, params: np.ndarray) -> float:
        """Negative fidelity with the target |0⟩ state."""
        param_binds = {p: v for p, v in zip(self.circuit.parameters, params)}
        result = self.sampler.run(self.circuit, shots=1024, param_binds=param_binds).result()
        counts = result.get_counts()
        p0 = counts.get('0', 0) / 1024
        return -p0

    def train(self, maxiter: int = 50) -> np.ndarray:
        """Optimize circuit parameters to maximise fidelity using COBYLA."""
        initial = np.random.uniform(-np.pi, np.pi, size=len(self.circuit.parameters))
        optimizer = COBYLA(maxiter=maxiter)
        optimizer.set_options({'disp': False})
        result = optimizer.minimize(self.fidelity_loss, initial)
        return result.x

    def get_qnn(self) -> SamplerQNN:
        return self.qnn

__all__ = ["AutoencoderGen470QNN"]
