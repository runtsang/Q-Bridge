import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder used as a feature extractor."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
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

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    return AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout)

def train_autoencoder(model, data, epochs: int = 100, batch_size: int = 64,
                      lr: float = 1e-3, weight_decay: float = 0.0,
                      device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that operates on the latent representation."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class Kernel(nn.Module):
    """RBF kernel that can be applied element‑wise to two tensors."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridFunctionQuantum(torch.autograd.Function):
    """Differentiable bridge that forwards a tensor to a quantum callable."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_callable, shift: float):
        ctx.shift = shift
        ctx.quantum_callable = quantum_callable
        expectation = quantum_callable(inputs.detach().cpu().numpy())
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        quantum_callable = ctx.quantum_callable
        eps = 1e-3
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            pos = inputs.clone()
            pos[i] += eps
            exp_pos = quantum_callable(pos.detach().cpu().numpy())
            neg = inputs.clone()
            neg[i] -= eps
            exp_neg = quantum_callable(neg.detach().cpu().numpy())
            grad = (exp_pos - exp_neg) / (2 * eps)
            grad_inputs[i] = grad
        return grad_inputs * grad_output, None, None

class HybridQuantumHead(nn.Module):
    """Wrapper that exposes the quantum expectation as a PyTorch module."""
    def __init__(self, quantum_callable, shift: float = 0.0):
        super().__init__()
        self.quantum_callable = quantum_callable
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunctionQuantum.apply(x, self.quantum_callable, self.shift)

class QCNetExtended(nn.Module):
    """
    Hybrid binary classifier that chains an auto‑encoder, self‑attention,
    RBF kernel, and a quantum expectation head.
    """
    def __init__(self,
                 autoencoder: nn.Module,
                 attention: nn.Module,
                 quantum_callable,
                 shift: float = 0.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.attention = attention
        self.kernel = Kernel()
        # Learnable prototype in attention space
        self.prototype = nn.Parameter(torch.randn(attention.embed_dim))
        self.quantum_head = HybridQuantumHead(quantum_callable, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        latent = self.autoencoder.encode(x)
        # Self‑attention
        attn = self.attention(latent)
        # Kernel similarity to prototype
        k = self.kernel(attn, self.prototype.unsqueeze(0))
        # Quantum expectation head
        qout = self.quantum_head(k.squeeze(-1))
        # Binary probability vector
        return torch.cat([qout, 1 - qout], dim=-1)

__all__ = [
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "ClassicalSelfAttention",
    "Kernel",
    "HybridFunctionQuantum",
    "HybridQuantumHead",
    "QCNetExtended",
]
