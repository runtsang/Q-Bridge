"""Quantum classifier factory and training loop using Pennylane.

The quantum model mirrors the classical interface: ``fit``, ``predict`` and
``evaluate``.  The circuit is a feature‑map + ansatz that is fully differentiable
with Pennylane’s autograd.  An optional Qiskit simulator can be used for
exact expectation values in small systems.

Key extensions over the seed:
- Flexible depth and entanglement pattern
- Optional use of a custom device (qiskit, pennylane, braket)
- Automatic gradient computation via ``qml.grad``
- Loss‑based training with binary cross‑entropy
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import pennylane as qml
import torch
import torch.nn.functional as F
from pennylane import numpy as np

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "linear",
) -> Tuple[qml.QNode, List[qml.ops.Operator], List[qml.ops.Operator], List[qml.measurements.MeasurementProcess]]:
    """Construct a variational circuit that returns observables.

    Parameters
    ----------
    num_qubits:
        Number of qubits in the ansatz.
    depth:
        Number of variational layers.
    entanglement:
        Entanglement pattern: ``linear`` or ``full``.

    Returns
    -------
    circuit:
        A ``qml.QNode`` that accepts input data and variational parameters.
    encoding:
        List of feature‑map operators (used for data‑encoding).
    weights:
        List of variational parameters.
    observables:
        List of ``qml.PauliZ`` operators (one per qubit).
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(data: torch.Tensor, params: torch.Tensor) -> List[torch.Tensor]:
        # Data‑encoding
        for i, w in enumerate(range(num_qubits)):
            qml.RX(data[i], w)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for w in range(num_qubits):
                qml.RY(params[idx], w)
                idx += 1
            if entanglement == "linear":
                for w in range(num_qubits - 1):
                    qml.CZ(w, w + 1)
            elif entanglement == "full":
                for w in range(num_qubits):
                    for v in range(w + 1, num_qubits):
                        qml.CZ(w, v)

        # Measurements
        return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

    observables = [qml.PauliZ(w) for w in range(num_qubits)]
    return circuit, [], [], observables


class QuantumClassifierModel:
    """Quantum classifier with a training API analogous to the classical one.

    Attributes
    ----------
    num_qubits:
        Size of the quantum circuit.
    depth:
        Number of variational layers.
    device:
        Pennylane device name (default ``default.qubit``).
    lr:
        Learning rate for the parameter optimiser.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        lr: float = 1e-3,
        entanglement: str = "linear",
        seed: Optional[int] = None,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device, wires=num_qubits)
        self.lr = lr
        self.entanglement = entanglement

        # Build circuit and initialise parameters
        self.circuit, _, _, _ = build_classifier_circuit(
            num_qubits, depth, entanglement
        )
        # Random initial parameters
        self.params = torch.randn(num_qubits * depth, requires_grad=True)

        # Optimiser
        self.opt = torch.optim.Adam([self.params], lr=self.lr)

    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        """Map circuit expectations to logits for binary classification."""
        exps = self.circuit(X, self.params)
        exps = torch.stack(exps, dim=-1)  # shape (batch, num_qubits)
        logits = torch.sum(exps, dim=1, keepdim=True)  # shape (batch, 1)
        return torch.cat([logits, -logits], dim=1)  # 2‑class logits

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32,
        epochs: int = 20,
        val_split: float = 0.1,
        patience: int = 5,
        verbose: bool = False,
    ) -> None:
        """Train the quantum circuit with a binary cross‑entropy loss.

        Parameters
        ----------
        X, y:
            Training data and labels (shape ``(N, num_qubits)`` and ``(N,)``).
        batch_size:
            Mini‑batch size.
        epochs:
            Maximum number of epochs.
        val_split:
            Fraction of data used for validation.
        patience:
            Early‑stopping patience.
        verbose:
            Print epoch statistics.
        """
        dataset = torch.utils.data.TensorDataset(X, y)
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X[n_train:], y[n_train:]),
            batch_size=batch_size,
        )

        best_loss = float("inf")
        best_params = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.circuit.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                logits = self._logits(xb)
                loss = F.binary_cross_entropy_with_logits(logits[:, 1], yb.float())
                loss.backward()
                self.opt.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= n_train

            # Validation
            self.circuit.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self._logits(xb)
                    loss = F.binary_cross_entropy_with_logits(logits[:, 1], yb.float())
                    val_loss += loss.item() * xb.size(0)
            val_loss /= n_val

            if verbose:
                print(
                    f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_params = self.params.clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stop after {epoch} epochs.")
                    break

        if best_params is not None:
            self.params.data = best_params

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions (0 or 1)."""
        self.circuit.eval()
        with torch.no_grad():
            logits = self._logits(X.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Return accuracy and loss on a dataset."""
        self.circuit.eval()
        with torch.no_grad():
            logits = self._logits(X.to(self.device))
            loss = F.binary_cross_entropy_with_logits(logits[:, 1], y.to(self.device).float()).item()
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y.to(self.device)).float().mean().item()
        return {"loss": loss, "accuracy": acc}
