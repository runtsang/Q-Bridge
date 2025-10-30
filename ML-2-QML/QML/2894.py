import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from typing import List

algorithm_globals.random_seed = 42

class AutoencoderHybridNet:
    """Variational quantum auto‑encoder that combines a RealAmplitudes ansatz,
    a domain‑wall pattern, and a swap‑test reconstruction step."""
    def __init__(self,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 reps: int = 5) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler
        )

    def _build_circuit(self) -> None:
        """Constructs the full variational circuit."""
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(qr, cr)

        # Variational ansatz
        ansatz = RealAmplitudes(num_qubits, reps=self.reps)
        self.circuit.compose(ansatz, range(num_qubits), inplace=True)

        # Domain‑wall: flip a contiguous block of qubits to encode a pattern
        for i in range(self.trash_dim, self.trash_dim + self.latent_dim):
            self.circuit.x(i)

        # Swap‑test between latent and trash registers
        aux = num_qubits - 1
        self.circuit.h(aux)
        for i in range(self.trash_dim):
            self.circuit.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        self.circuit.h(aux)
        self.circuit.measure(aux, cr[0])

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Run the circuit with the supplied parameters and return the sampled
        expectation values.  The output shape matches the sampler's output_shape."""
        return self.qnn.forward(params)

def Autoencoder(latent_dim: int = 3,
                trash_dim: int = 2,
                reps: int = 5) -> AutoencoderHybridNet:
    """Factory that mirrors the classical helper."""
    return AutoencoderHybridNet(latent_dim=latent_dim, trash_dim=trash_dim, reps=reps)

def train_autoencoder_qml(net: AutoencoderHybridNet,
                          data: np.ndarray,
                          epochs: int = 100,
                          optimizer_cls=COBYLA,
                          **optimizer_kwargs) -> List[float]:
    """Gradient‑free optimisation of the quantum circuit parameters."""
    opt = optimizer_cls(**optimizer_kwargs)
    loss_history: List[float] = []

    def loss_fn(params: np.ndarray) -> float:
        preds = net.forward(params)
        return np.mean((preds - data) ** 2)

    for _ in range(epochs):
        params, loss, _ = opt.optimize(
            num_vars=len(net.circuit.parameters),
            objective_function=loss_fn,
            init_point=np.random.randn(len(net.circuit.parameters))
        )
        loss_history.append(loss)

    return loss_history

__all__ = ["Autoencoder", "AutoencoderHybridNet", "train_autoencoder_qml"]
