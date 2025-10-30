"""Quantum classifier that extends the original data‑uploading ansatz.

The implementation uses Pennylane to construct a variational circuit that mirrors the
classical interface.  It adds:
* configurable entanglement depth
* automatic parameter‑shift gradients
* support for different optimisers (Adam, SPSA, etc.)
* batch‑wise training with cross‑entropy loss
* easy retrieval of circuit parameters and metadata

The class is fully importable and can be used as a drop‑in replacement for the
original seed.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QuantumClassifierModel:
    """
    Variational quantum classifier with a data‑uploading ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int, default 2
        Number of variational layers.
    device : str, default 'default.qubit'
        Pennylane device name.
    shots : int, default 1024
        Number of measurement shots per evaluation.
    opt_name : str, default 'adam'
        Optimiser name supported by Pennylane (e.g. 'adam','spsa', 'qng').
    lr : float, default 0.01
        Optimiser learning rate.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        device: str = "default.qubit",
        shots: int = 1024,
        opt_name: str = "adam",
        lr: float = 0.01,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device_name = device
        self.shots = shots
        self.opt_name = opt_name
        self.lr = lr

        # Variational parameters
        theta_init = np.random.uniform(0, 2 * np.pi, size=num_qubits * depth)
        self.theta = torch.nn.Parameter(
            torch.tensor(theta_init, dtype=torch.float32, requires_grad=True)
        )

        # Build the circuit
        self.dev = qml.device(device, wires=num_qubits, shots=shots)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # Optimiser
        if opt_name == "adam":
            self.optimizer = optim.Adam([self.theta], lr=self.lr)
        elif opt_name == "spsa":
            self.optimizer = qml.optim.SPSA(maxiter=1)
        else:
            raise ValueError(f"Unsupported optimiser: {opt_name}")

        # Metadata
        self.encoding = list(range(num_qubits))
        self.weight_sizes = [self.theta.numel()]
        self.observables = [qml.PauliZ(i) for i in range(num_qubits)]

    # ------------------------------------------------------------------ #
    #  Circuit definition
    # ------------------------------------------------------------------ #

    def _circuit(self, *params) -> List[float]:
        """Variational circuit with data‑encoding and entangling layers."""
        # Unpack parameters
        data = params[: self.num_qubits]
        theta = params[self.num_qubits :]

        # Data‑encoding
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RX(data[i], wires=wire)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for wire in range(self.num_qubits):
                qml.RY(theta[idx], wires=wire)
                idx += 1
            for wire in range(self.num_qubits - 1):
                qml.CZ(wire, wire + 1)

        # Measure expectation values of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels for a batch of inputs."""
        self.eval()
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities for a batch of inputs."""
        self.eval()
        outputs = []
        for x in X:
            out = self.qnode(*self.theta, *x)
            outputs.append(out.detach().numpy())
        logits = np.array(outputs)
        # Convert logits to probabilities
        return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        """Train the variational circuit using cross‑entropy loss."""
        self.train()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb_np = xb.numpy()
                # Forward pass
                preds = []
                for x in xb_np:
                    preds.append(self.qnode(*self.theta, *x))
                preds = torch.stack(preds)
                # Two‑class logits: sum of expectations and its negative
                logits = torch.stack([torch.sum(preds, dim=1), torch.sum(-preds, dim=1)], dim=1)
                loss = loss_fn(logits, yb)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            if verbose:
                print(f"Epoch {epoch:03d} – loss: {epoch_loss:.4f}")

    def eval(self) -> None:
        """Set circuit to evaluation mode."""
        # Pennylane QNodes are stateless; mode is implicit
        pass

    def train(self) -> None:
        """Set circuit to training mode."""
        pass

    def get_params(self) -> np.ndarray:
        """Return the current variational parameters."""
        return self.theta.detach().cpu().numpy()

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return (encoding, weight_sizes, observables)."""
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifierModel"]
