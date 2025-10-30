"""Quantum regression model that integrates a variational encoder, a kernel ansatz,
and a linear read‑out.  The model mirrors the classical hybrid above but
operates entirely on quantum states."""
from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
# 1. Data generation – complex superposition states
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    and a target label y = sin(2θ)·cos(φ).
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
    return states, labels.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a complex state vector and a target."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 3. Kernel ansatz – a list of parameterised gates
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """
    Encodes two input vectors x and y into a quantum state by
    applying a sequence of gates and then un‑computing the second vector.
    """
    def __init__(self, func_list: list[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """
    Quantum kernel evaluated via a fixed TorchQuantum ansatz.
    The kernel value is the absolute overlap of the final state with |0…0⟩.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 4. Variational regression circuit
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """
    A generic variational layer that applies a random circuit followed by
    trainable single‑qubit rotations on each wire.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class QModel(tq.QuantumModule):
    """
    Full regression model: encodes the input state, applies a variational
    layer, evaluates a quantum kernel, and reads out a scalar.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps classical features onto the computational basis
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLayer(num_wires)
        self.kernel = Kernel()
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode classical data
        self.encoder(qdev, state_batch)

        # Variational processing
        self.q_layer(qdev)

        # Kernel evaluation between the processed state and a stored basis
        # (here we reuse the last state as a trivial basis for illustration)
        kernel_vals = self.kernel(state_batch, state_batch)
        features = kernel_vals.unsqueeze(-1)  # shape (batch, 1)

        return self.head(features).squeeze(-1)


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QLayer",
    "QModel",
]
