import torch
from torch import nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Classical differentiable sigmoid head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridHead(nn.Module):
    """Linear layer followed by HybridFunction."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class SamplerQNN(nn.Module):
    """Classical neural network that mimics a sampler‑style QNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class QuantumHybridAutoencoder(nn.Module):
    """Classical autoencoder with a quantum‑inspired latent layer and hybrid head."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0.0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        # Quantum‑inspired latent transformation
        self.latent_reduce = nn.Linear(latent_dim, 2)
        self.latent_qnn = SamplerQNN()
        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
        # Hybrid head for final output
        self.head = HybridHead(input_dim)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_q = self.latent_qnn(self.latent_reduce(z))
        recon = self.decode(z_q)
        return self.head(recon)
