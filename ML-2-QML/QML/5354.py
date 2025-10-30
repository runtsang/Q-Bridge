from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumCircuit
from typing import Iterable, List, Sequence

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Quantum dataset that returns state vectors and scalar labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
    def __len__(self): return len(self.states)
    def __getitem__(self, index: int):
        return {"states": torch.tensor(self.states[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

class SamplerQNN(tq.QuantumModule):
    """Parameterized quantum circuit that outputs a probability distribution."""
    def __init__(self):
        super().__init__()
        self.qc = QuantumCircuit(2)
        self.qc.ry(tq.Param(0), 0)
        self.qc.ry(tq.Param(1), 1)
        self.qc.cx(0, 1)
        self.qc.ry(tq.Param(2), 0)
        self.qc.ry(tq.Param(3), 1)
        self.qc.cx(0, 1)
        self.qc.ry(tq.Param(4), 0)
        self.qc.ry(tq.Param(5), 1)
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.qc(qdev)
        probs = tq.measure_all(qdev)
        return probs

class QLayer(tq.QuantumModule):
    """Variational layer consisting of a random circuit and trainable rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

class HybridRegressionModel(tq.QuantumModule):
    """Quantum regression model that combines encoding, variational layer, sampler, and a classical head."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = QLayer(num_wires)
        self.sampler = SamplerQNN()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires + 2, 1)  # 2 from sampler log-probs
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        samp_probs = self.sampler(qdev)  # (bsz, 4) probability distribution
        log_probs = torch.log(samp_probs + 1e-12)[:, :2]  # use first two logits as features
        features = torch.cat([self.measure(qdev), log_probs], dim=-1)
        return self.head(features).squeeze(-1)

class FastEstimator:
    """Estimator that evaluates quantum circuit expectations over parameter grids."""
    def __init__(self, circuit: tq.QuantumDevice):
        self.circuit = circuit
    def evaluate(self,
                 observables: Iterable,
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            qdev = self.circuit.copy()
            qdev.assign_parameters(params)
            row = [qdev.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data",
           "SamplerQNN", "QLayer", "FastEstimator"]
