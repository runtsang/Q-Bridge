"""Hybrid quantum autoencoder with swap‑test encoding, domain‑wall preprocessing, and kernel evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence, List, Tuple
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

# --------------------------------------------------------------------------- #
# 1. RBF kernel (quantum‑side copy)
# --------------------------------------------------------------------------- #
class RBFKernel:
    """Simple RBF kernel for Gram‑matrix construction."""
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return np.exp(-self.gamma * np.dot(diff, diff))

# --------------------------------------------------------------------------- #
# 2. Helper to cast data
# --------------------------------------------------------------------------- #
def _as_array(data: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data

# --------------------------------------------------------------------------- #
# 3. Quantum autoencoder circuit
# --------------------------------------------------------------------------- #
class HybridAutoencoderQNN:
    """Quantum autoencoder built from a RealAmplitudes ansatz, swap‑test, and domain‑wall."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, kernel_gamma: float = 1.0):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.kernel = RBFKernel(gamma=kernel_gamma)
        self.circuit = self._build_circuit()
        # SamplerQNN for easy integration with QML primitives
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the full autoencoder circuit."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz for encoding
        ansatz = RealAmplitudes(num_qubits=self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        circuit.barrier()

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    # ----------------------------------------------------------------------- #
    # 4. Encode / decode helpers
    # ----------------------------------------------------------------------- #
    def encode(self, params: Sequence[float]) -> np.ndarray:
        """Return the statevector after binding parameters and running the circuit."""
        bound = self.circuit.assign_parameters(params, inplace=False)
        return Statevector(bound).state

    # ----------------------------------------------------------------------- #
    # 5. Kernel matrix
    # ----------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([[self.kernel(x, y) for y in b] for x in a])

    # ----------------------------------------------------------------------- #
    # 6. Evaluation of observables
    # ----------------------------------------------------------------------- #
    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for each parameter set."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(params, inplace=False)
            state = Statevector(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ----------------------------------------------------------------------- #
    # 7. Training routine (COBYLA on simple reconstruction loss)
    # ----------------------------------------------------------------------- #
    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 50,
        shots: int = 1024,
    ) -> List[float]:
        """Optimise the circuit parameters to minimise reconstruction error."""
        optimizer = COBYLA(steps=100, disp=False)
        history: List[float] = []

        def loss_fn(params: np.ndarray) -> float:
            loss = 0.0
            for x in data:
                bound = self.circuit.assign_parameters(params, inplace=False)
                state = Statevector(bound)
                # simple fidelity with |0...0> as reconstruction target
                loss += 1 - state.probabilities()[0]
            return loss / len(data)

        params0 = np.random.rand(len(self.circuit.parameters))
        for _ in range(epochs):
            params = optimizer.minimize(loss_fn, params0, bounds=[(0, 2 * np.pi)] * len(params0))
            history.append(loss_fn(params))
            params0 = params
        return history

__all__ = ["HybridAutoencoderQNN"]
