import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import qiskit

class QuantumSelfAttention:
    """Quantum self‑attention circuit matching the reference."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3*i], i)
            circuit.ry(rotation_params[3*i+1], i)
            circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            circuit.crx(entangle_params[i], i, i+1)
        circuit.measure(self.qr, self.cr)
        return circuit
    def run(self, backend, rotation_params, entangle_params, shots=1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = backend.run(circuit, shots=shots)
        return job.result().get_counts(circuit)

class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 use_quantum_latent: bool = False):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_quantum_latent = use_quantum_latent

class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with a quantum encoder and classical decoder."""
    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        self.config = config
        self.use_quantum_latent = config.use_quantum_latent
        self.simulator = AerSimulator()
        # Classical encoder (used only when not using quantum latent)
        if not self.use_quantum_latent:
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)
        # Self‑attention block
        self.attention = QuantumSelfAttention(n_qubits=4)
        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _quantum_latent_sample(self, inputs: np.ndarray):
        """Generate a latent vector by running a small variational circuit."""
        latents = []
        for inp in inputs:
            # Use first latent_dim entries as rotation angles
            angles = inp[:self.config.latent_dim]
            qc = QuantumCircuit(self.config.latent_dim)
            for i, ang in enumerate(angles):
                qc.rx(ang, i)
                qc.ry(ang, i)
                qc.rz(ang, i)
            ansatz = RealAmplitudes(self.config.latent_dim, reps=1)
            qc.compose(ansatz, inplace=True)
            result = self.simulator.run(qc).result()
            state = Statevector(result.get_statevector())
            # Expectation values of Pauli Z per qubit
            expz = [state.expectation_value(np.array([0,0,1,0])) for _ in range(self.config.latent_dim)]
            latents.append(expz)
        return np.array(latents)

    def encode(self, inputs: torch.Tensor):
        if not self.use_quantum_latent:
            return self.encoder(inputs)
        else:
            inp_np = inputs.cpu().numpy()
            latent_np = self._quantum_latent_sample(inp_np)
            latent = torch.as_tensor(latent_np, dtype=torch.float32, device=inputs.device)
            # Quantum self‑attention (placeholder)
            _ = self.attention.run(self.simulator, np.zeros(12), np.zeros(3), shots=1024)
            return latent

    def decode(self, latents: torch.Tensor):
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor):
        return self.decode(self.encode(inputs))

def HybridAutoencoderFactory(input_dim: int,
                            latent_dim: int = 32,
                            hidden_dims: tuple[int, int] = (128, 64),
                            dropout: float = 0.1,
                            use_quantum_latent: bool = False):
    cfg = HybridAutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout, use_quantum_latent)
    return HybridAutoencoder(cfg)

def train_hybrid_autoencoder_qml(model: HybridAutoencoder,
                                 data: torch.Tensor,
                                 *,
                                 epochs: int = 100,
                                 batch_size: int = 64,
                                 lr: float = 1e-3,
                                 weight_decay: float = 0.0,
                                 device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: np.ndarray | torch.Tensor):
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
