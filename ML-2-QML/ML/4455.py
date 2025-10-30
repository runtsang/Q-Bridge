"""FraudDetectionHybrid: Classical pipeline combining autoencoder, optional convolution, and a lightweight regressor.

The module uses the photonic‑style layer construction from the original FraudDetection seed,
but replaces the linear‑ReLU‑Tanh stack with a configurable autoencoder followed by a small
feed‑forward network.  An optional 2‑D convolutional filter can be inserted to handle
image‑like feature maps.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Sequence, Tuple, List

# Import helper modules from the same package (assumed to be available)
from Autoencoder import Autoencoder, AutoencoderConfig
from Conv import Conv
from EstimatorQNN import EstimatorQNN


def _tensorify(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection pipeline.

    Parameters
    ----------
    autoencoder_cfg : AutoencoderConfig
        Configuration for the autoencoder encoder/decoder.
    classifier_hidden : Sequence[int], optional
        Hidden layer sizes for the final regression network.
    use_conv : bool, default False
        Whether to prepend a 2‑D convolutional filter to the input.
    conv_kernel : int, default 2
        Kernel size for the convolutional filter.
    conv_threshold : float, default 0.0
        Threshold used by the convolutional filter.
    use_estimator : bool, default False
        If True, replace the simple feed‑forward regressor with the
        EstimatorQNN network defined in the reference seed.
    """

    def __init__(
        self,
        autoencoder_cfg: AutoencoderConfig,
        classifier_hidden: Sequence[int] = (8, 4),
        *,
        use_conv: bool = False,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        use_estimator: bool = False,
    ) -> None:
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = Conv()(kernel_size=conv_kernel, threshold=conv_threshold)
        else:
            self.conv = None

        # Autoencoder
        self.autoencoder = Autoencoder(
            input_dim=autoencoder_cfg.input_dim,
            latent_dim=autoencoder_cfg.latent_dim,
            hidden_dims=autoencoder_cfg.hidden_dims,
            dropout=autoencoder_cfg.dropout,
        )

        # Classifier
        if use_estimator:
            self.classifier = EstimatorQNN()
        else:
            layers: List[nn.Module] = []
            in_dim = autoencoder_cfg.latent_dim
            for h in classifier_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.Tanh())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.classifier = nn.Sequential(*layers)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional convolutional filter."""
        if self.conv is None:
            return x
        # Conv expects a 2‑D array per sample; reshape accordingly.
        # Here we assume x shape (N, C, H, W). If not, we reshape to (N, 1, k, k).
        if x.dim() == 2:
            # treat each row as flattened 2‑D patch
            k = int(self.conv.kernel_size)
            x = x.view(-1, 1, k, k)
        return torch.tensor(
            [self.conv.run(sample.cpu().numpy()) for sample in x]
        ).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: preprocess → encode → classify."""
        x = self._preprocess(x)
        latent = self.autoencoder.encode(x)
        return self.classifier(latent)

    def fit(
        self,
        X: Iterable[float] | torch.Tensor,
        y: Iterable[float] | torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> List[float]:
        """Simple training loop for the encoder + classifier."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X = _tensorify(X).to(device)
        y = _tensorify(y).unsqueeze(1).to(device)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

        return history

    def predict(self, X: Iterable[float] | torch.Tensor) -> torch.Tensor:
        """Return fraud probability for each sample."""
        self.eval()
        with torch.no_grad():
            X = _tensorify(X)
            return self.forward(X).squeeze(-1)

__all__ = ["FraudDetectionHybrid"]
