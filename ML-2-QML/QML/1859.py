"""Quantum autoencoder with swap‑test fidelity loss and COBYLA optimizer.

Provides:
- QuantumAutoencoder: builds a circuit with RealAmplitudes encoder and decoder.
- train_quantum_autoencoder: trains the QNN to minimize reconstruction error.

"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RealAmplitudes, StronglyEntanglingLayers
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

# Build autoencoder circuit
def build_quantum_autoencoder_circuit(input_dim: int, latent_dim: int, reps: int = 2):
    qr = QuantumRegister(input_dim + latent_dim, name='q')
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)

    # Encoder: RealAmplitudes on input qubits
    qc.append(RealAmplitudes(input_dim, reps=reps), range(input_dim))

    # Swap test between input and latent qubits
    aux = input_dim + latent_dim
    qc.add_register(QuantumRegister(1, name='aux'))
    qc.h(aux)
    for i in range(latent_dim):
        qc.cswap(aux, i, input_dim + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Decoder: RealAmplitudes on latent qubits
    qc.append(RealAmplitudes(latent_dim, reps=reps), range(input_dim, input_dim + latent_dim))

    return qc

# Quantum autoencoder object
class QuantumAutoencoder:
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 2, backend_name: str = "statevector_simulator"):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        self.circuit = build_quantum_autoencoder_circuit(input_dim, latent_dim, reps)
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=input_dim,
            sampler=self.sampler,
        )
        self.backend = Aer.get_backend(backend_name)

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Return the reconstructed statevector for given parameters."""
        self.circuit.set_parameters(dict(zip(self.circuit.parameters, params)))
        result = execute(self.circuit, self.backend, shots=0).result()
        state = result.get_statevector()
        return state

    def fidelity(self, input_state: np.ndarray, recon_state: np.ndarray) -> float:
        """Compute fidelity between two statevectors."""
        return np.abs(np.dot(np.conj(input_state), recon_state)) ** 2

# Training loop using COBYLA
def train_quantum_autoencoder(
    autoencoder: QuantumAutoencoder,
    training_states: np.ndarray,
    *,
    maxiter: int = 200,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Train the autoencoder to minimize reconstruction error on a set of statevectors."""
    initial_params = np.random.randn(len(autoencoder.circuit.parameters))
    def cost(params):
        total = 0.0
        for state in training_states:
            autoencoder.circuit.set_parameters(dict(zip(autoencoder.circuit.parameters, params)))
            result = execute(autoencoder.circuit, autoencoder.backend, shots=0).result()
            recon = result.get_statevector()
            total += 1.0 - autoencoder.fidelity(state, recon)
        return total / len(training_states)
    opt = COBYLA()
    opt.set_options({"maxiter": maxiter, "tol": tolerance})
    best_params, _ = opt.optimize(len(initial_params), cost, initial_params)
    return best_params

__all__ = ["QuantumAutoencoder", "train_quantum_autoencoder", "build_quantum_autoencoder_circuit"]
