"""Quantum regression model with depth‑controlled variational circuit.

The original seed implemented a single random layer followed by two
single‑qubit rotations.  In this extension the circuit consists of
several repeat cycles of:
    1) a parameterized entanglement layer (CX gates between adjacent
       qubits) and
    2) a layer of RX/RZ rotations whose parameters are trainable.
The depth is user‑configurable, allowing a systematic study of the
expressivity vs. trainability trade‑off.  After the variational
circuit the model measures all qubits in the Pauli‑Z basis and feeds
the resulting expectation values into a lightweight classical head.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    """
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping the synthetic quantum regression data.
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


class EntangleLayer(tq.QuantumModule):
    """Entanglement layer that applies CX gates between neighboring wires."""

    def __init__(self, wires: list[int]):
        super().__init__()
        self.wires = wires

    def forward(self, qdev: tq.QuantumDevice):
        for i in range(len(self.wires) - 1):
            tq.CX()(qdev, wires=[self.wires[i], self.wires[i + 1]])


class QLayer(tq.QuantumModule):
    """
    Variational layer with user‑defined depth.

    Parameters
    ----------
    num_wires :
        Number of qubits in the device.
    depth :
        Number of entanglement + rotation cycles.
    """

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.depth = depth
        self.entangle = EntangleLayer(list(range(num_wires)))
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for _ in range(self.depth):
            self.entangle(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.rz(qdev, wires=wire)


class QModel(tq.QuantumModule):
    """
    Quantum regression model with configurable circuit depth.

    Parameters
    ----------
    num_wires :
        Number of qubits in the device.
    depth :
        Depth of the variational layer.
    """

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = QLayer(num_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid quantum‑classical model.

        Parameters
        ----------
        state_batch :
            Batch of complex state vectors of shape ``(N, 2**D)``.

        Returns
        -------
        torch.Tensor :
            Predicted values of shape ``(N,)``.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def predict(
        self,
        X: torch.Tensor,
        batch_size: int = 256,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Run inference on a large batch of quantum states.

        Parameters
        ----------
        X :
            Input tensor of shape ``(N, 2**D)``.
        batch_size :
            Number of samples processed per GPU/CPU chunk.
        device :
            Target device.  If ``None`` the model's device is used.

        Returns
        -------
        torch.Tensor :
            Predicted values of shape ``(N,)``.
        """
        self.eval()
        device = device or next(self.parameters()).device
        X = X.to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = X[i : i + batch_size]
                preds.append(self(batch))
        return torch.cat(preds)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
