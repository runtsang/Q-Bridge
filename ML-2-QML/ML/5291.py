import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Local imports from the quantum module
from qml_code import QuantumCircuitWrapper, HybridLayer, FraudLayerParams

# --------------------------------------------------------------------------- #
#  Classical components
# --------------------------------------------------------------------------- #

class ClassicalAutoencoder(nn.Module):
    """A lightweight auto‑encoder that compresses CNN features into a low‑dimensional latent space."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64)):
        super().__init__()
        # Encoder
        encoder = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], latent_dim)
        ]
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = [
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        ]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class ClassicalSamplerQNN(nn.Module):
    """Simple neural sampler that produces a probability distribution over two modes."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

class FraudDetectionLayer(nn.Module):
    """Linear + activation + scaling layer inspired by photonic fraud‑detection circuits."""
    def __init__(self, params: FraudLayerParams, clip: bool = False):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)

        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

class FraudDetectionModel(nn.Sequential):
    """Sequential stack of fraud‑detection layers followed by a final linear classifier."""
    def __init__(self, input_params: FraudLayerParams, layers: list[FraudLayerParams]):
        modules = [FraudDetectionLayer(input_params, clip=False)]
        modules += [FraudDetectionLayer(l, clip=True) for l in layers]
        modules.append(nn.Linear(2, 1))
        super().__init__(*modules)

# --------------------------------------------------------------------------- #
#  Hybrid quantum‑classical network
# --------------------------------------------------------------------------- #

class HybridQCNet(nn.Module):
    """
    CNN → Auto‑encoder → Sampler → Quantum head → Fraud‑detection stack → Probabilities.
    Designed for binary image classification but can be extended to other modalities.
    """
    def __init__(self, input_shape: tuple[int, int, int] = (3, 32, 32)):
        super().__init__()
        # 1. Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the dimension after convolution to initialise the auto‑encoder
        dummy = torch.zeros(1, *input_shape)
        conv_out = self.feature_extractor(dummy)
        conv_dim = conv_out.shape[1]

        self.autoencoder = ClassicalAutoencoder(input_dim=conv_dim, latent_dim=32)

        # 2. Latent sampler
        self.sampler = ClassicalSamplerQNN()

        # 3. Quantum head
        self.quantum_head = HybridLayer(n_qubits=2, shots=200, shift=np.pi/2)

        # 4. Fraud detection stack
        # Dummy parameters – replace with real photonic parameters for a production system
        zero_params = FraudLayerParams(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        self.fraud_model = FraudDetectionModel(zero_params, [zero_params, zero_params])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feat = self.feature_extractor(x)

        # Auto‑encoder latent representation
        latent = self.autoencoder.encode(feat)

        # Sample from latent distribution (classical sampler)
        # Use the first two latent dimensions as the sampler input
        sampled = self.sampler(latent[:, :2])

        # Quantum head consumes a single parameter (first latent dimension)
        quantum_out = self.quantum_head(latent[:, :1])

        # Fraud detection stack operates on the quantum output and the first latent dimension
        fraud_input = torch.cat([quantum_out, latent[:, :1]], dim=-1)
        logits = self.fraud_model(fraud_input)

        # Binary probability distribution
        probs = torch.softmax(logits, dim=-1)
        return probs

__all__ = [
    "ClassicalAutoencoder",
    "ClassicalSamplerQNN",
    "FraudDetectionLayer",
    "FraudDetectionModel",
    "HybridQCNet",
]
