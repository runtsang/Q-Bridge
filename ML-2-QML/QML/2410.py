import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

class UnifiedAutoencoderHybrid:
    """Quantum autoencoder that uses a variational circuit and swap‑test reconstruction."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2, shots: int = 1024) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.backend,
        )
        self.optimizer = COBYLA()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Parameterised encoding layer
        qc.append(RealAmplitudes(self.num_latent + self.num_trash, reps=5), range(self.num_latent + self.num_trash))
        qc.barrier()
        # Swap‑test for reconstruction fidelity
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the quantum circuit with the given parameters."""
        self.qnn.set_weights(params)
        result = self.qnn.forward(np.array([params]))
        return result[0]

    def train(self, data: np.ndarray, epochs: int = 20, lr: float = 0.1) -> list[float]:
        """Train the quantum autoencoder by minimizing MSE between input and reconstructed output."""
        num_params = len(self.qnn.weight_params)
        init_params = np.random.randn(num_params)
        loss_history: list[float] = []

        def loss_fn(params: np.ndarray) -> float:
            self.qnn.set_weights(params)
            recon = []
            for _ in data:
                out = self.qnn.forward(np.array([params]))
                recon.append(out[0])
            recon = np.array(recon)
            loss = np.mean((data - recon) ** 2)
            return loss

        self.optimizer.minimize(loss_fn, init_params)
        return loss_history
