import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, Optional

class Sine(nn.Module):
    """Periodic activation mimicking quantum expectation."""
    def forward(self, x):
        return torch.sin(x)

class HybridBinaryClassifier(nn.Module):
    """
    A binary classifier that can operate in two modes:

    * ``classical`` – a dense head with a sigmoid activation.
    * ``quantum_approx`` – a lightweight neural approximation of a quantum
      expectation layer. The approximation uses a sine activation to mimic
      the periodicity of quantum circuits.

    The class also provides convenient helpers for data augmentation,
    early‑stopping training, and ONNX export.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Tuple[int,...] = (120, 84),
        mode: str = "classical",
        use_dropout: bool = True,
    ) -> None:
        super().__init__()
        assert mode in ("classical", "quantum_approx"), "mode must be 'classical' or 'quantum_approx'"
        self.mode = mode
        self.use_dropout = use_dropout

        # Shared feature extractor
        self.fc1 = nn.Linear(in_features, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

        # Classical head
        self.classical_head = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

        # Quantum‑approx head
        self.quantum_head = nn.Sequential(
            nn.Linear(1, 1),
            Sine(),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        if self.mode == "classical":
            out = self.classical_head(x)
        else:
            out = self.quantum_head(x)
        # Return probabilities for both classes
        return torch.cat((out, 1 - out), dim=-1)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def augment_batch(batch: torch.Tensor, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple data augmentation: random horizontal flips and rotations.
        """
        if torch.rand(1).item() > 0.5:
            batch = torch.flip(batch, dims=[-1])
        if torch.rand(1).item() > 0.5:
            batch = torch.rot90(batch, k=1, dims=[-2, -1])
        return batch, probs

    def train_with_early_stopping(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        patience: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Train the model with early stopping based on validation loss.
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0

        self.to(device)
        for epoch in range(epochs):
            self.train()
            for batch, labels in train_loader:
                batch, labels = batch.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 0.0
            self.eval()
            with torch.no_grad():
                for batch, labels in val_loader:
                    batch, labels = batch.to(device), labels.to(device)
                    outputs = self(batch)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), "best_model.pt")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def export_onnx(self, input_shape: Tuple[int,...], file_path: str = "model.onnx") -> None:
        """
        Export the model to ONNX format.
        """
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(self, dummy_input, file_path, opset_version=11,
                          input_names=["input"], output_names=["output"])

__all__ = ["HybridBinaryClassifier"]
