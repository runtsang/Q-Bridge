import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Classical backbone (mirrors the original Autoencoder.py)
from.Autoencoder import AutoencoderNet, AutoencoderConfig

# Quantum helper – a lightweight feature extractor
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler

class QuantumFeatureExtractor(nn.Module):
    """
    Uses a RealAmplitudes ansatz to map a classical latent vector
    into a quantum state.  The statevector is returned as a real tensor
    suitable for downstream classical layers.
    """
    def __init__(self, num_qubits: int, reps: int = 3) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.circuit = RealAmplitudes(num_qubits, reps=reps)
        self.sampler = Sampler()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor of shape (batch, num_qubits)
            Classical latent vector to be encoded.
        Returns
        -------
        torch.Tensor of shape (batch, 2**num_qubits)
            Real part of the sampled statevectors.
        """
        batch, _ = latent.shape
        outputs = []
        for vec in latent:
            qc = self.circuit.assign_parameters(vec.tolist())
            result = self.sampler.run(qc)
            state = result.result().get_statevector()
            outputs.append(state.real)
        return torch.tensor(outputs, dtype=torch.float32, device=latent.device)

class HybridAutoencoder(nn.Module):
    """
    Combines the classical AutoencoderNet with an optional quantum feature
    extractor.  When ``use_quantum`` is True, the latent representation
    produced by the encoder is passed through a quantum circuit before
    decoding.  This design keeps the encoder/decoder architecture identical
    to the original, while providing a plug‑in for quantum experiments.
    """
    def __init__(self,
                 config: AutoencoderConfig,
                 use_quantum: bool = False,
                 quantum_reps: int = 3) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(config)
        self.decoder = AutoencoderNet(config)
        self.use_quantum = use_quantum
        if use_quantum:
            self.qfe = QuantumFeatureExtractor(config.latent_dim, reps=quantum_reps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        if self.use_quantum:
            latent = self.qfe(latent)
        return self.decoder(latent)

def train_hybrid(model: nn.Module,
                 data: torch.Tensor,
                 *,
                 epochs: int = 100,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 device: torch.device | None = None) -> list[float]:
    """
    Simple reconstruction training loop that works for both classical and
    hybrid models.  Returns a list of epoch‑wise MSE losses.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "train_hybrid", "AutoencoderConfig", "AutoencoderNet"]
