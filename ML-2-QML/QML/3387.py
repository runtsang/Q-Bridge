"""Hybrid quantum regression that marries a data‑encoding kernel ansatz
with a parameter‑shaped random layer and a linear read‑out head.  The
module mirrors the classical API, enabling direct substitution in
experiments targeting both back‑ends.

Key design points
-----------------
*  The `KernelAnsatz` implements the same functional form as the
   classical RBF kernel but in a quantum circuit.  It accepts two
   input tensors and applies a reversible sequence of rotations.
*  `HybridRegressionModel` uses a GeneralEncoder to embed the
   raw state, runs the kernel ansatz, measures a single Pauli‑Z
   expectation per wire, and feeds the measurement vector to a
   classical linear head.
*  All hyper‑parameters (number of wires, depth of the random layer,
   encoder choice) are exposed as constructor arguments, allowing
   systematic scaling studies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----- data generation ----------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form
    cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray
        Shape (samples, 2**num_wires) with dtype complex.
    labels : np.ndarray
        Shape (samples,) with a non‑linear target function.
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

# ----- dataset -------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns complex quantum states and real targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.states.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----- quantum kernel ansatz ------------------------------------------------ #
class KernelAnsatz(tq.QuantumModule):
    """Encodes two classical vectors into a quantum circuit and then
    un‑encodes the difference, mirroring a quantum kernel.
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset states to |0...0⟩
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Un‑encode y with negative parameters
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed 4‑wire kernel ansatz implementing a simple product of Ry gates."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernelAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return absolute overlap of the final state with |0...0⟩
        return torch.abs(self.q_device.states.view(-1)[0])

# ----- hybrid model --------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that encodes inputs, applies a random
    layer, measures Pauli‑Z, and feeds the expectation vector to a
    classical linear head.

    Parameters
    ----------
    num_wires : int
        Number of qubits to use for encoding.
    random_layer_depth : int
        Number of random two‑qubit gates in the random layer.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            # A depth‑controlled random layer of CNOT and single‑qubit rotations
            self.random_layer = tq.RandomLayer(
                n_ops=depth * 2,  # rough mapping to depth
                wires=list(range(n_wires)),
                param_dict={"cx": {"prob": 0.5}},
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, q_dev: tq.QuantumDevice) -> None:
            self.random_layer(q_dev)
            for w in range(self.n_wires):
                self.rx(q_dev, wires=w)
                self.ry(q_dev, wires=w)

    def __init__(self, num_wires: int = 4, random_layer_depth: int = 30):
        super().__init__()
        self.n_wires = num_wires
        # Use a GeneralEncoder that maps a real vector into the computational basis
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self._QLayer(num_wires, random_layer_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        q_dev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(q_dev, state_batch)
        self.q_layer(q_dev)
        features = self.measure(q_dev)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "Kernel", "KernelAnsatz"]
