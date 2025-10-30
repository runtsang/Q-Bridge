import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.circuit import ParameterVector

class HybridAutoencoder:
    """Quantum autoencoder with a variational encoder and a classical decoder."""
    def __init__(self, latent_dim=4, num_trash=2, reps=5, shots=1024):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.shots = shots
        self.sampler = StatevectorSampler()
        self._build_circuit()

    def _ansatz(self, num_qubits):
        return RealAmplitudes(num_qubits, reps=self.reps)

    def _build_circuit(self):
        num_latent, num_trash = self.latent_dim, self.num_trash
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        self.circuit.barrier()
        aux = num_latent + 2 * num_trash
        self.circuit.h(aux)
        for i in range(num_trash):
            self.circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def encode(self, params: np.ndarray) -> np.ndarray:
        """Map classical parameters to a quantum state via the ansatz."""
        if len(params)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch")
        bound = self.circuit.bind_parameters(params)
        state = Statevector.from_instruction(bound)
        return state.data

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Simple linear decoder mapping latent vector to reconstruction."""
        # Learnable weights would be learned in a full training loop
        w = np.random.randn(z.size, 784) * 0.01
        return z @ w

    def forward(self, params: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(params))

    def evaluate(self, observables, parameter_sets):
        """Evaluate expectation values for a list of parameter sets."""
        results = []
        for params in parameter_sets:
            state = Statevector.from_instruction(self.circuit.bind_parameters(params))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def optimize_params(self, target: np.ndarray, *,
                        init_params: np.ndarray | None = None,
                        max_evals: int = 100):
        """Optimize the circuit parameters to reconstruct a single target vector."""
        if init_params is None:
            init_params = np.random.randn(len(self.circuit.parameters))
        def loss(p):
            pred = self.forward(p)
            return np.mean((pred - target) ** 2)
        optimizer = COBYLA()
        params, _ = optimizer.optimize(len(init_params), loss, init_params)
        self.params = params
        return params

__all__ = ["HybridAutoencoder"]
