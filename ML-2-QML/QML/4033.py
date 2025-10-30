"""Quantum hybrid regression model incorporating a SamplerQNN style circuit.

Features:
- Amplitude‑encoded data via GeneralEncoder.
- RandomLayer with adaptive number of operations.
- A classical RY+RZ rotation block.
- A SamplerQNN inspired circuit for probability estimation.
- Head: linear mapping from measurement outcomes to a scalar regression output.

The module preserves the same public API as the classical counterpart, making it trivial to benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states as in the original seed but with more diversity."""
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
    """Quantum dataset returning complex state tensors and scalar labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNN(tq.QuantumModule):
    """Quantum sampler network modeled after the classical SamplerQNN.

    It consists of a parameterized 2‑qubit circuit with Ry rotations followed by a CX gate,
    mimicking the structure of the original Qiskit example.
    """
    def __init__(self):
        super().__init__()
        self.params = tq.ParameterVector("qnn_input", 2)
        self.weights = tq.ParameterVector("qnn_weight", 4)
        self.circuit = tq.Circuit(2)
        self.circuit.ry(self.params[0], 0)
        self.circuit.ry(self.params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.circuit(qdev)
        # Return probability amplitudes of |00> and |01>
        probs = tq.MeasureAll(tq.PauliZ)(qdev)
        return probs[:, :2]

class HybridRegression(QuantumModule):
    """Hybrid quantum regression model with a SamplerQNN block."""
    def __init__(self, num_wires: int, use_sampler: bool = False):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._quantum_block()
        self.use_sampler = use_sampler
        self.sampler = SamplerQNN() if use_sampler else None
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires + (2 if use_sampler else 0), 1)

    def _quantum_block(self) -> tq.QuantumModule:
        """Random layer followed by trainable RY+RZ rotations."""
        class _Block(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=40, wires=list(range(n_wires)))
                self.ry = tq.RY(has_params=True, trainable=True)
                self.rz = tq.RZ(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                for w in range(self.n_wires):
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
        return _Block(num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.use_sampler:
            # Embed batch into sampler via a 2‑qubit circuit
            sampler_dev = tq.QuantumDevice(n_wires=2, bsz=bsz, device=state_batch.device)
            # Simple encoding: use first two features as rotation angles
            angles = state_batch[:, :2]
            sampler_dev.ry(angles[:, 0], 0)
            sampler_dev.ry(angles[:, 1], 1)
            sampler_out = self.sampler(sampler_dev)
            features = torch.cat([features, sampler_out], dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
