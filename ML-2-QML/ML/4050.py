"""Hybrid classifier that fuses classical convolution and a quantum variational circuit.

The class QuantumHybridClassifier is a drop‑in replacement for the original
`QuantumClassifierModel`.  It first runs a 2×2 convolutional filter on each
sample, concatenates the resulting scalar with the raw features, encodes the
combined vector into RX angles, executes a Qiskit circuit built by the QML
module, and finally maps the expectation values to logits with a linear
head.  The module is fully compatible with PyTorch, NumPy and the
Qiskit simulator.
"""

import torch
import torch.nn as nn
import numpy as np
import qiskit
from typing import Tuple

# Local imports
from Conv import Conv
from qml_module import build_classifier_circuit

class QuantumHybridClassifier(nn.Module):
    """
    PyTorch module that combines a 2×2 convolutional feature extractor,
    a parameterised quantum circuit and a classical linear head.

    Parameters
    ----------
    num_features : int
        Number of classical features per sample.
    num_qubits : int
        Number of qubits used in the quantum circuit.
    depth : int
        Depth of the variational ansatz.
    conv_kernel : int, default 2
        Size of the convolutional kernel.
    conv_threshold : float, default 0.0
        Threshold used by the classical Conv filter.
    backend_name : str, default 'qasm_simulator'
        Qiskit backend to run the circuit.
    shots : int, default 1024
        Number of shots for the quantum simulation.
    """
    def __init__(self,
                 num_features: int,
                 num_qubits: int,
                 depth: int,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 backend_name: str = "qasm_simulator",
                 shots: int = 1024):
        super().__init__()
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.depth = depth
        self.conv = Conv()  # classical convolutional filter
        # Build quantum circuit and store metadata
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_classifier_circuit(
            num_qubits, depth)
        self.backend_name = backend_name
        self.shots = shots

        # Classical head that maps the quantum expectation vector to logits
        self.head = nn.Linear(num_qubits, 2)

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical features into rotation angles in [0, 2π].
        """
        # Normalise to [0, 2π]
        max_val = x.max() if x.max() > 0 else 1.0
        return (x / max_val) * 2 * np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, num_features)
            Raw input features.

        Returns
        -------
        torch.Tensor of shape (batch, 2)
            Logits for binary classification.
        """
        batch_size = x.shape[0]
        # Classical convolutional feature extraction
        images = x.view(batch_size, 1, 2, 2)
        conv_out = torch.tensor([self.conv.run(img.squeeze(0).cpu().numpy()) for img in images],
                                device=x.device).unsqueeze(-1)  # shape (batch, 1)

        # Combine with original features
        combined = torch.cat([x, conv_out], dim=1)  # shape (batch, num_features+1)
        # Truncate if necessary
        if combined.shape[1] > self.num_qubits:
            combined = combined[:, :self.num_qubits]

        # Encode into circuit parameters
        angles = self._encode_features(combined)

        # Prepare parameter binds for Qiskit
        param_binds = []
        for i in range(batch_size):
            bind = {param: angles[i, j].item() for j, param in enumerate(self.encoding_params)}
            # initialise variational parameters to zero (trainable via Qiskit later)
            for w in self.weight_params:
                bind[w] = 0.0
            param_binds.append(bind)

        # Execute the circuit on the simulator
        job = qiskit.execute(self.circuit,
                             backend=qiskit.Aer.get_backend(self.backend_name),
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()

        # Compute expectation values for each observable
        expectations = []
        for obs in self.observables:
            exp = result.get_expectation_value(obs, self.circuit)
            expectations.append(exp)
        expectations = torch.tensor(expectations, dtype=torch.float32, device=x.device)

        # Classical head
        logits = self.head(expectations)
        return logits

# Dataset utilities (borrowed from reference pair 2)
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Binary labels for classification
    labels = (y > 0).astype(np.int64)
    return x, labels.astype(np.int64)

class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.long),
        }

__all__ = ["QuantumHybridClassifier", "HybridDataset", "generate_superposition_data"]
