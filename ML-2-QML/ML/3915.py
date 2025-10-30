import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Configuration and helper utilities
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderHybridConfig:
    """
    Configuration for the hybrid autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    latent_dim : int, default=32
        Size of the latent representation.
    hidden_dims : Tuple[int,...], default=(128, 64)
        Sizes of the hidden layers in the encoder/decoder.
    dropout : float, default=0.1
        Dropout probability applied after each hidden layer.
    use_quantum : bool, default=False
        Whether to include a quantum encoder/decoder in the model.
    quantum_latent_dim : int, default=3
        Number of qubits used for the quantum latent space.
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_quantum: bool = False
    quantum_latent_dim: int = 3


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Classical MLP autoencoder
# --------------------------------------------------------------------------- #

class ClassicalAutoencoder(nn.Module):
    """
    A standard multilayer perceptron autoencoder.

    The encoder and decoder share a mirrored architecture defined by
    `hidden_dims`.  Dropout is applied after each hidden layer.
    """

    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
# Fraud‑detection style parameterised layers
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """
    Parameters describing a single photonic‑style layer.

    The design is inspired by the original fraud‑detection example and
    intentionally mirrors the structure of a 2‑neuron linear block
    followed by a custom activation and scaling.
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
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
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def fraud_autoencoder(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    output_dim: int = 1,
) -> nn.Sequential:
    """
    Construct a sequential PyTorch model that mimics the structure of a
    photonic fraud‑detection circuit.

    The first layer is un‑clipped; subsequent layers are clipped to keep
    parameters within a safe range for hardware‑like simulation.
    """
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, output_dim))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Hybrid Autoencoder: combines classical and fraud‑style layers
# --------------------------------------------------------------------------- #

class AutoencoderHybrid(nn.Module):
    """
    Unified autoencoder that can operate in three modes:

    1. Classical MLP only (``use_quantum=False``).
    2. Fraud‑detection style layers only (``use_quantum=False`` and
       ``use_fraud=True``).
    3. A hybrid of both classical decoder and a quantum encoder
       (``use_quantum=True``).

    The constructor decides which sub‑module to instantiate based on the
    supplied configuration.
    """

    def __init__(self, config: AutoencoderHybridConfig, *, fraud_params: Optional[List[FraudLayerParameters]] = None) -> None:
        super().__init__()
        self.config = config
        self.use_quantum = config.use_quantum

        if self.use_quantum:
            # The quantum part is only a placeholder; the actual circuit
            # is defined in the QML module.  Here we keep a dummy
            # linear map to preserve API compatibility.
            self.quantum_encoder = nn.Linear(config.input_dim, config.latent_dim)
            self.decoder = ClassicalAutoencoder(config).decoder
        else:
            # Classical only
            self.autoencoder = ClassicalAutoencoder(config)

        # Fraud‑style autoencoder branch (optional)
        if fraud_params is not None:
            self.fraud_branch = fraud_autoencoder(
                fraud_params[0], fraud_params[1:], output_dim=config.input_dim
            )
        else:
            self.fraud_branch = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self.quantum_encoder(inputs)
        return self.autoencoder.encode(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self.decoder(latents)
        return self.autoencoder.decode(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.fraud_branch is not None:
            # Run fraud branch first, then pass through the chosen decoder
            fraud_output = self.fraud_branch(inputs)
            return self.decode(fraud_output)
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #

def train_autoencoder_hybrid(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """
    Standard reconstruction training loop for the hybrid autoencoder.

    Parameters
    ----------
    model : AutoencoderHybrid
        The model to train.
    data : torch.Tensor
        Training data of shape (N, input_dim).
    epochs : int, default=100
        Number of full passes through the data.
    batch_size : int, default=64
        Mini‑batch size.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    weight_decay : float, default=0.0
        L2 regularisation strength.
    device : torch.device, optional
        Device on which to perform training.  Defaults to CUDA if available.

    Returns
    -------
    List[float]
        History of mean reconstruction loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridConfig",
    "FraudLayerParameters",
    "fraud_autoencoder",
    "train_autoencoder_hybrid",
    "ClassicalAutoencoder",
]
