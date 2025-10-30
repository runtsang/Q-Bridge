"""
Hybrid binary classifier – classical implementation.

The module contains:
  * `HybridFunction` – a differentiable sigmoid head.
  * `Hybrid` – a dense layer that replaces the quantum head.
  * `Autoencoder`, `AutoencoderConfig`, and `train_autoencoder` – a
    lightweight MLP auto‑encoder from the reference seed.
  * `build_classifier_circuit` – a feed‑forward classifier factory
    mirroring the quantum helper.
  * `EstimatorQNN` – a tiny regression head (optional).
  * `HybridBinaryClassifierML` – the main classifier that can
    optionally prepend a classical auto‑encoder and use either a
    classical or a hybrid (quantum‑style) head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

# --------------------------------------------------------------------------- #
# 1. Classical Auto‑Encoder (from Reference 2)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the MLP auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Helper that guarantees a float32 tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# 2. Classical Classifier Factory (from Reference 3)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 3. Classical Hybrid Function (from Reference 1)
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# --------------------------------------------------------------------------- #
# 4. Optional Regression Head (from Reference 4)
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """Tiny fully‑connected regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


# --------------------------------------------------------------------------- #
# 5. Convolutional Backbone (shared with the QML variant)
# --------------------------------------------------------------------------- #
class ConvBackbone(nn.Module):
    """Standard 2‑D CNN backbone."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.drop2(x)


# --------------------------------------------------------------------------- #
# 6. Main Hybrid Binary Classifier – Classical Path
# --------------------------------------------------------------------------- #
class HybridBinaryClassifierML(nn.Module):
    """
    A fully‑classical binary classifier that optionally inserts a
    pre‑training auto‑encoder and can switch between a classical
    or a quantum‑style head.

    Parameters
    ----------
    use_autoencoder : bool
        If True, a classical MLP auto‑encoder is inserted between the
        backbone and the head.
    use_hybrid_head : bool
        If True, the final head uses the `Hybrid` module (a dense + sigmoid)
        which mimics the quantum expectation head.
    autoencoder_config : AutoencoderConfig | None
        Configuration for the auto‑encoder when `use_autoencoder` is True.
    classifier_depth : int
        Depth of the fully‑connected classifier when `use_hybrid_head` is False.
    """
    def __init__(
        self,
        *,
        use_autoencoder: bool = False,
        use_hybrid_head: bool = True,
        autoencoder_config: Optional[AutoencoderConfig] = None,
        classifier_depth: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        feature_dim = 55815  # output of the flattened backbone
        self.fc1 = nn.Linear(feature_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            config = autoencoder_config or AutoencoderConfig(input_dim=120)
            self.autoencoder = Autoencoder(
                input_dim=120,
                latent_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        else:
            self.autoencoder = None

        if use_hybrid_head:
            self.head = Hybrid(1, shift=0.0)
        else:
            # Build a classical feed‑forward classifier
            self.head, _, _, _ = build_classifier_circuit(1, classifier_depth)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(inputs)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.use_autoencoder:
            x = self.autoencoder.encode(x)

        logits = self.head(x)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs


# --------------------------------------------------------------------------- #
# 7. Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "train_autoencoder",
    "build_classifier_circuit",
    "HybridFunction",
    "Hybrid",
    "EstimatorQNN",
    "ConvBackbone",
    "HybridBinaryClassifierML",
]
