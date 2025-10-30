import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable, Sequence

# ------------------------------------------------------------------
# Classical autoencoder
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    return AutoencoderNet(cfg)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
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

# ------------------------------------------------------------------
# Self‑attention layer
# ------------------------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor) -> torch.Tensor:
        Q = inputs @ W_q
        K = inputs @ W_k
        scores = torch.softmax(Q @ K.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

def SelfAttention(embed_dim: int) -> ClassicalSelfAttention:
    return ClassicalSelfAttention(embed_dim)

# ------------------------------------------------------------------
# Fraud‑style neural network
# ------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Hybrid kernel class
# ------------------------------------------------------------------
class HybridQuantumKernel:
    """
    Classical preprocessing pipeline that compresses raw data with an
    autoencoder, refines it via self‑attention, and feeds the result
    into a fraud‑style fully‑connected network.  The processed
    representation can be supplied to a quantum kernel evaluator
    defined in the QML module.
    """
    def __init__(
        self,
        auto_cfg: AutoencoderConfig,
        attention_embed: int,
        fraud_input: FraudLayerParameters,
        fraud_layers: Sequence[FraudLayerParameters],
    ):
        self.autoencoder = Autoencoder(auto_cfg)
        self.attention = SelfAttention(attention_embed)
        self.fraud_net = build_fraud_detection_program(fraud_input, fraud_layers)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        # Identity projections for an initial demonstration; learnable in practice
        W_q = torch.eye(z.shape[-1], dtype=torch.float32)
        W_k = torch.eye(z.shape[-1], dtype=torch.float32)
        att = self.attention(z, W_q, W_k)
        return self.fraud_net(att)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute a classical RBF kernel over the processed features.
        """
        feats_a = torch.stack([self.preprocess(x) for x in a])
        feats_b = torch.stack([self.preprocess(y) for y in b])
        diff = feats_a.unsqueeze(1) - feats_b.unsqueeze(0)
        dist_sq = (diff ** 2).sum(-1)
        return np.exp(-dist_sq.cpu().numpy())

__all__ = ["HybridQuantumKernel", "AutoencoderConfig", "Autoencoder", "train_autoencoder",
           "SelfAttention", "FraudLayerParameters", "build_fraud_detection_program"]
