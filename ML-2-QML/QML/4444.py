"""Quantum implementation of the hybrid kernel method.

The class `HybridKernelMethod` is a thin wrapper around a TorchQuantum
ansatz that evaluates a quantum kernel.  It also contains a
parameterised sampler circuit and a quantum regression model that
mirrors the classical counterparts.  The API matches the classical
module so that both can be swapped at runtime.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn
from typing import Sequence

# ---------------------------------------------------------------------------

class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel with optional sampler and fully‑connected layer.

    Parameters
    ----------
    n_wires : int
        Number of qubits used for the kernel circuit.
    use_sampler : bool
        When ``True`` a simple parameterised sampler circuit is attached
        to the module.  The sampler is a drop‑in replacement for the
        classical `SamplerQNN`.
    """
    def __init__(self, n_wires: int = 4, use_sampler: bool = False) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Build a simple RY‑ansatz that mirrors the classical RBF kernel
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = self._build_sampler()

    def _ansatz_forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode two classical vectors and apply the ansatz."""
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap of the two encoded states."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self._ansatz_forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix for two collections of samples."""
        return np.array([[float(self(x, y)) for y in b] for x in a])

    # -----------------------------------------------------------------------
    #  Optional sampler circuit
    # -----------------------------------------------------------------------
    def _build_sampler(self) -> tq.QuantumModule:
        """Return a simple parameterised sampler that mimics the classical one."""
        class Sampler(tq.QuantumModule):
            def __init__(self) -> None:
                super().__init__()
                self.q_device = tq.QuantumDevice(n_wires=2)
                self.encoder = tq.RY(has_params=True, trainable=True)
                self.cx = tq.CX()
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
                # encode two parameters
                self.encoder(qdev, wires=[0], params=qdev.states[:, 0])
                self.encoder(qdev, wires=[1], params=qdev.states[:, 1])
                self.cx(qdev, wires=[0, 1])
                return self.measure(qdev)

        return Sampler()

    # -----------------------------------------------------------------------
    #  Quantum regression model
    # -----------------------------------------------------------------------
    class QModel(tq.QuantumModule):
        """Quantum regression network inspired by the seed."""
        class QLayer(tq.QuantumModule):
            def __init__(self, num_wires: int):
                super().__init__()
                self.n_wires = num_wires
                self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice) -> None:
                self.random_layer(qdev)
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
            self.q_layer = self.QLayer(num_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(num_wires, 1)

        def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
            bsz = state_batch.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
            self.encoder(qdev, state_batch)
            self.q_layer(qdev)
            features = self.measure(qdev)
            return self.head(features).squeeze(-1)

    # -----------------------------------------------------------------------
    #  Utility functions mirroring the classical dataset
    # -----------------------------------------------------------------------
    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

class RegressionDataset(tq.QuantumModule):
    """Dataset that returns quantum state tensors and a scalar target."""
    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = HybridKernelMethod.generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "HybridKernelMethod",
    "RegressionDataset",
    "generate_superposition_data",
]
