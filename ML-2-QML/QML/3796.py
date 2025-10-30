import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuantumDecoder:
    """Encodes a latent vector into a swap‑test based quantum decoder circuit.

    The circuit is parameterised by the latent vector and outputs a 2‑dimensional
    measurement that can be fed into a classical post‑processing network.
    """
    def __init__(self,
                 latent_dim: int,
                 num_trash: int = 2,
                 reps: int = 5,
                 seed: int = 42) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        self.latent_params = [Parameter(f"latent_{i}") for i in range(latent_dim)]
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.latent_params,
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.latent_dim + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Apply the latent rotations
        for i, p in enumerate(self.latent_params):
            qc.ry(p, i)

        # Variational ansatz on the latent + trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash,
                                reps=self.reps,
                                seed=self.latent_dim + self.num_trash + 1)
        qc.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)

        # Swap‑test with auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of latent vectors."""
        outputs = []
        for vec in latent:
            param_map = {p: float(vec[i]) for i, p in enumerate(self.latent_params)}
            out = self.qnn.forward(param_map)
            outputs.append(out)
        return np.vstack(outputs)

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        return self.forward(latent)

class EstimatorNN:
    """Lightweight classical regressor that post‑processes the quantum decoder output."""
    def __init__(self) -> None:
        from torch import nn
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        import torch
        x = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            out = self.net(x)
        return out.numpy()

class HybridAutoencoder:
    """Combines the QuantumDecoder with a classical EstimatorNN for full reconstruction."""
    def __init__(self, latent_dim: int) -> None:
        self.decoder = QuantumDecoder(latent_dim)
        self.postprocess = EstimatorNN()

    def forward(self, latent: np.ndarray) -> np.ndarray:
        """Decode a batch of latent vectors into reconstructions."""
        dec_out = self.decoder.forward(latent)          # shape (batch, 2)
        recon = self.postprocess.forward(dec_out)      # shape (batch, 1)
        return recon

__all__ = ["HybridAutoencoder", "QuantumDecoder", "EstimatorNN"]
