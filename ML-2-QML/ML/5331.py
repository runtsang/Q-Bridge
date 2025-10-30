"""
EstimatorQNNGen: hybrid classical‑quantum estimator.

This class fuses the feed‑forward regression, CNN feature extractor,
and quantum expectation head from the seed examples.  It can be
used as a drop‑in replacement for the original EstimatorQNN and
supports both regression and multi‑class classification.

Features
--------
* CNN feature extractor (two convolutional layers + max‑pool)
* Fully connected projection to a vector of size *n_qubits*
* Quantum expectation head implemented by QuantumCircuitWrapper
* Classical sigmoid head as a fallback
* Configurable backend, shot count, parameter‑shift value, and
  number of output classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The quantum primitives are defined in the sibling module.
# They are imported lazily to avoid importing qiskit when the
# quantum head is disabled.
try:
    from.qml_code import HybridQuantum
except Exception:  # pragma: no cover
    HybridQuantum = None


class HybridClassicalHead(nn.Module):
    """Fallback sigmoid head used when the quantum part is disabled."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


class EstimatorQNNGen(nn.Module):
    """
    Hybrid estimator that combines a CNN, a fully‑connected projection,
    and a quantum expectation head.

    Parameters
    ----------
    num_classes : int, default 1
        Number of output targets. 1 for regression, >1 for classification.
    use_quantum : bool, default True
        If ``True`` the quantum head is used; otherwise a classical sigmoid
        head is applied.
    n_qubits : int, default 4
        Number of qubits in the parameterised quantum circuit.
    backend : str | qiskit.providers.Backend, default "qasm_simulator"
        Backend used for the quantum circuit.
    shots : int, default 1024
        Number of shots used to estimate the expectation value.
    shift : float, default np.pi/2
        Shift value for the parameter‑shift gradient rule.
    """

    def __init__(
        self,
        num_classes: int = 1,
        use_quantum: bool = True,
        n_qubits: int = 4,
        backend: str | None = "qasm_simulator",
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.shift = shift

        # Classical feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Projection to the quantum feature space
        self.fc = nn.Linear(16 * 7 * 7, n_qubits)

        # Classical fallback head
        self.classical_head = HybridClassicalHead(n_qubits, shift=shift)

        # Quantum head
        if use_quantum:
            if HybridQuantum is None:
                raise ImportError(
                    "Quantum module 'EstimatorQNN__gen220_qml' is missing."
                )
            self.quantum_head = HybridQuantum(
                n_qubits=n_qubits,
                backend=backend,
                shots=shots,
                shift=shift,
            )
        else:
            self.quantum_head = None

        # Final classification layer (if needed)
        if num_classes > 1:
            self.classifier = nn.Linear(n_qubits, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Predictions of shape (B, num_classes).
        """
        # Feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Projection to quantum space
        x = self.fc(x)

        # Quantum or classical head
        if self.use_quantum and self.quantum_head is not None:
            out = self.quantum_head(x)
        else:
            out = self.classical_head(x)

        # Final classification layer
        if self.classifier is not None:
            out = self.classifier(out)
        return out


__all__ = ["EstimatorQNNGen"]
