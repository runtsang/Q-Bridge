import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from.FraudDetection import FraudLayerParameters, build_fraud_detection_program

class FraudDetectionModel(nn.Module):
    """
    Classical fraud detection model with preprocessing, noise augmentation,
    and a hybrid loss that can be paired with a quantum circuit.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: list[FraudLayerParameters],
                 noise_std: float = 0.01,
                 device: torch.device | str | None = None):
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.noise_std = noise_std

        # Build core sequential model
        self.model = build_fraud_detection_program(input_params, layers).to(self.device)

        # Pre‑processing
        self.scaler = StandardScaler()
        self.register_buffer("scaler_mean", torch.zeros(2, device=self.device))
        self.register_buffer("scaler_scale", torch.ones(2, device=self.device))

    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit the StandardScaler on training data and store the parameters
        as buffers for use in the forward pass.
        """
        self.scaler.fit(X)
        self.scaler_mean.copy_(torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device))
        self.scaler_scale.copy_(torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with preprocessing, noise augmentation and the core model.
        """
        # Pre‑process
        x = (x - self.scaler_mean) / self.scaler_scale

        # Augmentation
        noise = torch.randn_like(x, device=self.device) * self.noise_std
        x = x + noise

        # Core model
        return self.model(x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute a hybrid loss: binary cross‑entropy + L2 regularisation
        on all linear layers.  The regulariser encourages the model to
        stay within the physically meaningful parameter range seen in the seed.
        """
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        l2 = torch.tensor(0.0, device=self.device)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                l2 += torch.norm(module.weight, p=2) + torch.norm(module.bias, p=2)
        return bce + 1e-4 * l2

    def train_step(self, optimizer: torch.optim.Optimizer,
                   batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Single optimisation step.  The method is intentionally lightweight
        so that the training loop can be customised externally.
        """
        optimizer.zero_grad()
        logits = self(batch)
        loss = self.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        return loss
