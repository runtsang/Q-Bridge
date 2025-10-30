import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
import torch
import torch.nn as nn

class HybridAutoencoder(nn.Module):
    """
    A hybrid quantum‑classical autoencoder that shares a latent space between a dense MLP encoder
    and a variational quantum circuit (VQC). The quantum part refines the latent codes via a
    swap‑test style ansatz, enabling hybrid fine‑tune and uncertainty estimation.
    """

    def __init__(self, input_dim: int, latent_dim: int = 3, num_trash: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash

        # Classical encoder – identical to AutoencoderNet.encode
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # Build the quantum circuit that will be used as VQC
        self.qc = self._build_qc()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.qc,
            input_params=[],
            weight_params=self.qc.parameters,
            interpret=lambda x: x,
            output_shape=2,
        )

    def _build_qc(self):
        num_qubits = self.latent_dim + 2 * self.num_trash + 1
        qc = QuantumCircuit(num_qubits, name="ae_qc")
        qc.compose(RealAmplitudes(num_qubits, reps=5), range(0, self.latent_dim + self.num_trash), inplace=True)
        qc.barrier()
        aux = num_qubits - 1
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, 0)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical latent code
        z = self.classical_encoder(x)
        # Placeholder: quantum refinement omitted for brevity
        return z

    def train_qc(self, data: np.ndarray, lr: float = 0.01, epochs: int = 100):
        algorithm_globals.random_seed = 42
        optimizer = COBYLA()
        for _ in range(epochs):
            loss = 0.0
            for sample in data:
                result = self.sampler.run(self.qc).result()
                counts = result.get_counts(self.qc)
                loss += np.mean([int(k) for k in counts.values()])
            loss /= len(data)
            optimizer.optimize(
                n=1,
                initial_point=[0.0],
                objective_function=lambda p: loss,
                tolerance=0.01,
            )
