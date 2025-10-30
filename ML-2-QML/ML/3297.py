import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes

class HybridAutoencoderFCL(nn.Module):
    """
    Classical auto‑encoder with an embedded quantum fully‑connected layer.
    The quantum layer contributes a single expectation value per sample
    which is concatenated to the classical latent vector before decoding.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        n_qubits: int = 2,
        shots: int = 200,
    ) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum fully‑connected layer (single‑qubit expectation)
        self.n_qubits = n_qubits
        self.shots = shots
        self.sampler = Sampler()
        self.qc_template = self._build_qc_template()

        # Decoder now expects one extra feature from the quantum layer
        decoder_layers = []
        in_dim = latent_dim + 1  # +1 for quantum output
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _build_qc_template(self) -> QuantumCircuit:
        """
        Builds a template circuit that accepts n_qubits rotation angles.
        The circuit prepares a state, measures Z on qubit 0 and returns
        an expectation value.
        """
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        # Parameterised Ry rotations
        for i in range(self.n_qubits):
            qc.ry(qiskit.circuit.Parameter(f"theta_{i}"), qr[i])
        qc.measure(qr[0], cr[0])
        return qc

    def _quantum_expectation(self, angles: np.ndarray) -> torch.Tensor:
        """
        Evaluate the expectation value of Z on qubit 0 given rotation angles.
        angles: (batch, n_qubits)
        Returns a tensor of shape (batch, 1).
        """
        batch, _ = angles.shape
        expectations = []
        for i in range(batch):
            bind = {f"theta_{j}": angles[i, j] for j in range(self.n_qubits)}
            job = self.sampler.run(self.qc_template, parameter_binds=[bind], shots=self.shots)
            result = job.result()
            counts = result.get_counts(self.qc_template)
            # Convert bitstring to integer (0 or 1)
            probs = np.array([counts.get('0', 0), counts.get('1', 0)]) / self.shots
            exp = 1.0 * probs[0] + (-1.0) * probs[1]  # Z expectation
            expectations.append(exp)
        return torch.tensor(np.array(expectations).reshape(-1, 1), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode → quantum layer → decode.
        """
        latent = self.encoder(x)
        # Use first n_qubits of latent as angles for quantum circuit
        angles = latent[:, :self.n_qubits].detach().cpu().numpy()
        quantum_out = self._quantum_expectation(angles).to(x.device)
        latent_with_q = torch.cat([latent, quantum_out], dim=1)
        reconstructed = self.decoder(latent_with_q)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

__all__ = ["HybridAutoencoderFCL"]
