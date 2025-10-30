import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition states of the form
    cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    The labels are a sinusoidal function of the angles used to construct the state.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset that exposes the quantum states as complex tensors and the target
    labels as real tensors.  The states are stored as ``torch.cfloat`` to preserve
    the phase information required for quantum measurements.
    """
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
    """
    Hybrid quantum‑classical regression model built on PennyLane.  The quantum circuit
    consists of a data‑encoding layer followed by a stack of entangling blocks and
    parameterised single‑qubit rotations.  The expectation values of Pauli‑Z are fed
    into a classical linear head to produce a scalar prediction.
    """
    def __init__(self, num_wires: int, entanglement: str = "circular", layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.entanglement = entanglement
        self.layers = layers

        # Define a parameterised quantum device
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        # Build a variational circuit
        def circuit(state, params):
            # State‑encoding via a rotation about the Y‑axis
            for w in range(num_wires):
                qml.RY(state[w], wires=w)

            # Variational layers
            for l in range(layers):
                for w in range(num_wires):
                    qml.RX(params[l, w, 0], wires=w)
                    qml.RZ(params[l, w, 1], wires=w)
                if entanglement == "circular":
                    for w in range(num_wires):
                        qml.CNOT(wires=[w, (w + 1) % num_wires])

            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        # Create a QNode that will be compiled by PennyLane
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        ``state_batch`` is a batch of complex vectors of shape ``(B, 2**N)``.
        The QNode expects a real vector of length ``N`` for the encoding, so we
        convert the complex amplitudes into a real angle representation.
        """
        # Convert each state to an angle vector by taking the argument of the
        # amplitude of the |1...1> component.  This is a toy encoding for
        # demonstration purposes.
        angle_batch = torch.angle(state_batch[:, -1]).unsqueeze(-1)
        # Broadcast to match the device's wire count
        angle_batch = angle_batch.repeat(1, self.num_wires)

        # Initialise parameters for the variational layers
        params = torch.randn(self.layers, self.num_wires, 2, requires_grad=True, device=state_batch.device)

        # Execute the QNode in a batched fashion
        expectations = self.qnode(angle_batch, params)

        # Pass through the classical head
        return self.head(expectations).squeeze(-1)

    def predict(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns the model output without the extra
        ``squeeze`` used in the training loop.
        """
        return self.forward(state_batch)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
