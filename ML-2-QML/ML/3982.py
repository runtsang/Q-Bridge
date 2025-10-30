import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ConvGen302(nn.Module):
    """Hybrid convolution–autoencoder filter with optional quantum mode."""
    def __init__(
        self,
        input_shape: Tuple[int, int],
        kernel_size: int = 2,
        use_quanv: bool = False,
        use_autoencoder: bool = False,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.use_quanv = use_quanv
        self.use_autoencoder = use_autoencoder
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not use_quanv and not use_autoencoder:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        else:
            self.conv = None

        if use_autoencoder:
            self.encoder = nn.Sequential(
                nn.Linear(kernel_size ** 2, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], kernel_size ** 2),
                nn.Tanh(),
            )
        else:
            self.encoder = None
            self.decoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quanv:
            raise NotImplementedError("Quantum mode must be called via.run_quanv()")
        elif self.use_autoencoder:
            B, C, H, W = x.shape
            pad = self.kernel_size // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant")
            patches = x_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
            patches = patches.contiguous().view(-1, self.kernel_size ** 2).to(self.device)
            encoded = self.encoder(patches)
            decoded = self.decoder(encoded).view(-1, 1, self.kernel_size, self.kernel_size)
            out = F.conv2d(x, decoded, bias=None, stride=1, padding=0)
            return out
        else:
            return self.conv(x)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        if not self.use_autoencoder:
            raise RuntimeError("Autoencoder mode not enabled")
        self.encoder.train()
        self.decoder.train()
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(self.device)
                B, C, H, W = batch.shape
                pad = self.kernel_size // 2
                batch_padded = F.pad(batch, (pad, pad, pad, pad), mode="constant")
                patches = batch_padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
                patches = patches.contiguous().view(-1, self.kernel_size ** 2).to(self.device)
                optimizer.zero_grad()
                recon = self.decoder(self.encoder(patches))
                loss = loss_fn(recon, patches)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * patches.size(0)
            epoch_loss /= len(loader.dataset)

    def run_quanv(
        self,
        data: np.ndarray,
        backend=None,
        shots: int = 100,
        threshold: float = 0.5,
    ) -> float:
        if not self.use_quanv:
            raise RuntimeError("Quanv mode not enabled")
        import qiskit
        from qiskit import Aer, execute
        from qiskit.circuit import Parameter
        n_qubits = self.kernel_size ** 2
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circ.rx(theta[i], i)
        circ.barrier()
        circ.measure_all()
        param_binds = []
        for val in data.reshape(-1, n_qubits):
            bind = {theta[i]: np.pi if v > threshold else 0.0 for i, v in enumerate(val)}
            param_binds.append(bind)
        job = execute(circ, backend or Aer.get_backend("qasm_simulator"), shots=shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(circ)
        total = sum(cnt * sum(int(bit) for bit in bitstring) for bitstring, cnt in counts.items())
        return total / (shots * n_qubits * len(param_binds))
