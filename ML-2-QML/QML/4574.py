import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates quantum‑encoded superposition states.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape ``(samples, 2**num_wires)``.
    labels : np.ndarray
        Real regression targets.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that emits complex state vectors and real labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegression(tq.QuantumModule):
    """
    Quantum‑hybrid regression model that mirrors the classical counterpart.

    The model performs:
    1. Feature encoding via a parameterised Ry‑chain (GeneralEncoder).
    2. A short random variational layer followed by single‑qubit rotations.
    3. Measurement of all qubits in the Pauli‑Z basis.
    4. A classical linear head that maps the expectation vector to a scalar prediction.
    """
    def __init__(self, num_wires: int, hidden: int = 8):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Real input vector of shape ``(batch, num_wires)``.

        Returns
        -------
        torch.Tensor
            Continuous regression prediction.
        """
        bsz = x.shape[0]
        qdev = QuantumDevice(n_wires=self.encoder.n_wires, bsz=bsz, device=x.device)
        # Encode classical data into a quantum state
        self.encoder(qdev, x)
        # Variational layer
        self.random(qdev)
        for w in range(self.encoder.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        # Extract expectation values
        features = self.measure(qdev)
        out = self.head(features).squeeze(-1)
        return out

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
