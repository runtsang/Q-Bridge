import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

# --------------------------------------------------------------------------- #
#  Classical auto‑encoder backbone
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight MLP auto‑encoder that feeds a quantum latent vector."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
#  Quantum encoder wrapper (Qiskit)
# --------------------------------------------------------------------------- #
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.opflow import StateFn
from qiskit.quantum_info import Statevector

class QuantumEncoder(nn.Module):
    """Variational quantum circuit that maps a classical vector to a quantum state."""
    def __init__(self, latent_dim: int, reps: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.reps = reps
        # Parameterized ansatz
        self.ansatz = RealAmplitudes(latent_dim, reps=reps, insert_barriers=True)

    def forward(self, classical_vec: torch.Tensor) -> torch.Tensor:
        """
        Encode a classical vector into a quantum state represented as a state vector.
        """
        # Create a quantum circuit and set parameters
        circuit = QuantumCircuit(self.latent_dim)
        # Classical vector is interpreted as angles for the ansatz
        circuit.compose(self.ansatz.bind_parameters(
            {f'θ_{i}': float(x) for i, x in enumerate(classical_vec)}
        ), inplace=True)

        # Simulate to get the statevector
        backend = AerSimulator(method='statevector')
        result = backend.run(circuit).result()
        sv = result.get_statevector(circuit)
        # Convert to a torch tensor
        return torch.tensor(sv, dtype=torch.complex64, device=classical_vec.device)

# --------------------------------------------------------------------------- #
#  Photonic fraud‑detection subnet (Strawberry Fields)
# --------------------------------------------------------------------------- #
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

class PhotonicFraudSubnet(nn.Module):
    """Wrapper around a Strawberry Fields photonic program."""
    def __init__(self, layer_params: List[dict], clip: bool = True) -> None:
        super().__init__()
        self.layer_params = layer_params
        self.clip = clip
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            for params in self.layer_params:
                self._apply_layer(q, params, clip=self.clip)
        return prog

    def _apply_layer(self, q, params: dict, *, clip: bool) -> None:
        BSgate(params['bs_theta'], params['bs_phi']) | (q[0], q[1])
        for i, phase in enumerate(params['phases']):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params['squeeze_r'], params['squeeze_phi'])):
            Sgate(r if not clip else max(-5.0, min(5.0, r)), phi) | q[i]
        BSgate(params['bs_theta'], params['bs_phi']) | (q[0], q[1])
        for i, phase in enumerate(params['phases']):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params['displacement_r'], params['displacement_phi'])):
            Dgate(r if not clip else max(-5.0, min(5.0, r)), phi) | q[i]
        for i, k in enumerate(params['kerr']):
            Kgate(k if not clip else max(-1.0, min(1.0, k))) | q[i]

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the photonic program on a 2‑mode input and return the expectation
        value of the photon number in the first mode as a fraud score.
        """
        # Convert torch tensor to numpy
        np_input = input_vec.detach().cpu().numpy()
        eng = sf.Engine('gaussian')
        result = eng.run(self.program, args={'0': np_input[0], '1': np_input[1]})
        # Photon number expectation in mode 0
        exp_num = result.expectation_value(sf.ops.N(0))
        return torch.tensor(exp_num, dtype=torch.float32, device=input_vec.device)

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class HybridAutoFraudEncoder(nn.Module):
    """
    Combines a classical auto‑encoder, a variational quantum encoder, and a
    photonic fraud‑detection subnet.  The forward pass returns both the
    reconstruction and a fraud score.
    """
    def __init__(
        self,
        cfg: AutoencoderConfig,
        quantum_reps: int = 3,
        photonic_layers: Optional[List[dict]] = None,
    ) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(cfg)
        self.quantum = QuantumEncoder(cfg.latent_dim, reps=quantum_reps)
        # If no photonic config provided, use a simple dummy layer
        if photonic_layers is None:
            photonic_layers = [{
                'bs_theta': 0.0,
                'bs_phi': 0.0,
                'phases': (0.0, 0.0),
               'squeeze_r': (0.0, 0.0),
               'squeeze_phi': (0.0, 0.0),
                'displacement_r': (0.0, 0.0),
                'displacement_phi': (0.0, 0.0),
                'kerr': (0.0, 0.0),
            }]
        self.fraud_subnet = PhotonicFraudSubnet(photonic_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch, input_dim)

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed input
        fraud_score : torch.Tensor
            Fraud probability / anomaly score from the photonic subnet
        """
        # Classical encoding
        latent_classical = self.autoencoder.encode(x)
        # Quantum encoding (state vector)
        latent_quantum = self.quantum(latent_classical)
        # For simplicity, collapse the quantum state to a real vector by taking real part
        latent_for_fraud = torch.real(latent_quantum)[:, :2]  # take first two modes
        fraud_score = self.fraud_subnet(latent_for_fraud)
        # Reconstruction
        reconstruction = self.autoencoder.decode(latent_classical)
        return reconstruction, fraud_score

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> List[float]:
        """
        Simple training loop that optimizes reconstruction loss and fraud score
        together.  The fraud score loss is added with a small weight.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, fraud = self(batch)
                recon_loss = mse_loss(recon, batch)
                fraud_loss = torch.mean(fraud)  # encourage higher fraud score for anomalies
                loss = recon_loss + 0.01 * fraud_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)
        return loss_history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "QuantumEncoder",
    "PhotonicFraudSubnet",
    "HybridAutoFraudEncoder",
]
