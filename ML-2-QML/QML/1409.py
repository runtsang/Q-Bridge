import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class HybridAutoencoder:
    """Variational quantum autoencoder with swap‑test fidelity loss."""
    def __init__(self, num_qubits: int, latent_dim: int, num_trash: int, reps: int = 3):
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self._build_circuit()

    def _build_circuit(self):
        """Construct the variational autoencoder circuit."""
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Encode ansatz
        ansatz = RealAmplitudes(self.num_qubits, reps=self.reps)
        self.circuit.compose(ansatz, range(self.num_qubits), inplace=True)

        # Swap‑test for fidelity
        aux = self.num_qubits
        self.circuit.h(aux)
        for i in range(self.num_trash):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def fidelity_loss(self, input_sv: Statevector) -> float:
        """Compute negative fidelity between input state and circuit output."""
        backend = Aer.get_backend("statevector_simulator")
        job = execute(self.circuit, backend, shots=1, parameter_binds={})
        result = job.result()
        output_sv = Statevector(result.get_statevector())
        fidelity = output_sv.fidelity(input_sv)
        return -fidelity

    def train(self, data: list[Statevector], epochs: int = 20, lr: float = 0.01):
        """Simple COBYLA optimizer to maximize fidelity on given dataset."""
        algorithm_globals.random_seed = 42
        opt = COBYLA()
        params = opt.get_initial_params(self.circuit.num_parameters)
        def objective(p):
            self.circuit.assign_parameters(p, inplace=True)
            loss = 0.0
            for sv in data:
                loss += self.fidelity_loss(sv)
            return loss / len(data)
        opt.minimize(objective, params, maxiter=epochs)
        self.circuit.assign_parameters(opt.x, inplace=True)

    def sample(self, num_shots: int = 1024) -> np.ndarray:
        """Sample measurement distribution from the circuit."""
        backend = Aer.get_backend("qasm_simulator")
        job = execute(self.circuit, backend, shots=num_shots)
        counts = job.result().get_counts()
        probs = np.zeros(2)
        for outcome, count in counts.items():
            probs[int(outcome)] = count / num_shots
        return probs

__all__ = ["HybridAutoencoder"]
