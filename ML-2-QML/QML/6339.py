from __future__ import annotations
import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩ and labels sin(2θ)cosφ."""
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
    """Dataset that pairs superposition states with their continuous labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalSamplerNet(nn.Module):
    """
    Classical sampler identical to the one in the ML module but self‑contained.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, input_dim),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, weight_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inp = self.input_net(x)
        inp = torch.softmax(inp, dim=-1)
        w = self.weight_net(x)
        return {"input_params": inp, "weight_params": w}

class HybridQRegressor(tq.QuantumModule):
    """
    Quantum regression model that uses a classical sampler to produce
    input and weight parameters for a variational circuit.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice, weight_params: torch.Tensor) -> None:
            self.random_layer(qdev)
            # weight_params shape (batch, 2*num_wires)
            rx_params = weight_params[:, :self.n_wires]
            ry_params = weight_params[:, self.n_wires:]
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire, params=rx_params[:, wire])
                self.ry(qdev, wires=wire, params=ry_params[:, wire])

    def __init__(self, num_wires: int, sampler_net: ClassicalSamplerNet | None = None):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.sampler_net = sampler_net or ClassicalSamplerNet()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: shape (batch, 2) – two‑dimensional classical input that drives
        the sampler. It is also used as the feature vector for the quantum encoder.
        """
        bsz = state_batch.shape[0]
        sampler_out = self.sampler_net(state_batch)
        weight_params = sampler_out["weight_params"]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev, weight_params)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

def SamplerQNN() -> HybridQRegressor:
    """
    Factory that returns a ready‑to‑train hybrid quantum regression model.
    """
    return HybridQRegressor(num_wires=2)

__all__ = ["HybridQRegressor", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
