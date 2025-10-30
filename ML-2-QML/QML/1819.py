"""Quantum regression model using Pennylane and a variational circuit."""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Same data generation as classical variant, but returns complex amplitude vectors."""
    # Construct basis states |0...0> and |1...1>
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a complex state vector and a real label."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Hybrid quantum‑classical regression model built on Pennylane.

    The architecture consists of:
      * A state‑encoding layer that prepares the input state.
      * A variational block with alternating RX/RZ rotations and CNOT entanglement.
      * Measurement of all qubits in the Z basis, yielding a real‑valued feature vector.
      * A classical linear head that maps the quantum features to a scalar prediction.
    """
    def __init__(self, num_wires: int, device: str = "default.qubit", shots: int = 1000):
        super().__init__()
        self.num_wires = num_wires
        self.device_str = device
        self.shots = shots

        # Variational parameters (learnable)
        self.params = nn.Parameter(torch.randn(num_wires * 2, dtype=torch.float64))

        # Quantum circuit definition
        dev = qml.device(device, wires=num_wires, shots=shots)
        self.qnode = qml.QNode(self._circuit, dev=dev, interface="torch")

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

    def _circuit(self, params: torch.Tensor, state: torch.Tensor):
        """Prepare the input state, apply variational layers, and measure."""
        # Convert torch tensor to numpy for state preparation
        state_np = state.detach().cpu().numpy()
        # State preparation – angle‑encoding
        for wire, amp in enumerate(state_np):
            theta = 2 * np.arccos(np.abs(amp))
            phi = np.angle(amp)
            qml.RY(theta, wires=wire)
            qml.RZ(phi, wires=wire)

        # Variational block: 2 layers of RX/RZ + CNOT entanglement
        for layer in range(2):
            for wire in range(self.num_wires):
                qml.RX(params[wire], wires=wire)
                qml.RZ(params[wire + self.num_wires], wires=wire)
            for wire in range(self.num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])

        # Return expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        features = []
        for i in range(bsz):
            feat = self.qnode(self.params, state_batch[i])
            features.append(feat)
        features = torch.stack(features, dim=0)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
