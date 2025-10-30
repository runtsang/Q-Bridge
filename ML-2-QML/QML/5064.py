"""Hybrid quantum convolutional regression with fraud detection."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from typing import Iterable, Sequence

# Quantum convolution filter
class QuanvCircuit:
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 0.0):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# Quantum kernel ansatz
class KernalAnsatz(tq.QuantumModule):
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# Classical regression head after quantum kernel
class QRegressionHead(tq.QuantumModule):
    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int,...] = (32, 16)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Photonic fraud detection program
class FraudLayerParameters:
    def __init__(self, bs_theta: float, bs_phi: float, phases: tuple[float, float],
                 squeeze_r: tuple[float, float], squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float], displacement_phi: tuple[float, float],
                 kerr: tuple[float, float]) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# Combined quantum hybrid model
class HybridConvolutionalRegressor(tq.QuantumModule):
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 gamma: float = 1.0,
                 regression_input_dim: int = 1,
                 fraud_input: FraudLayerParameters | None = None,
                 fraud_layers: list[FraudLayerParameters] | None = None):
        super().__init__()
        self.conv = QuanvCircuit(conv_kernel_size, shots=200, threshold=conv_threshold)
        self.kernel = QuantumKernel()
        self.regression = QRegressionHead(regression_input_dim)
        if fraud_input is None:
            fraud_input = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
            fraud_layers = []
        self.fraud_program = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (batch, 1, H, W)
        conv_outputs = []
        for img in images:
            kernel = img.squeeze().flatten().numpy()
            conv_outputs.append(self.conv.run(kernel))
        conv_outputs = torch.tensor(conv_outputs, dtype=torch.float32).unsqueeze(1)
        # Kernel similarity
        kernel_matrix = self.kernel(conv_outputs, conv_outputs)
        features = kernel_matrix.mean(dim=1)
        preds = self.regression(features.unsqueeze(1))
        # Fraud score placeholder: zero tensor
        fraud_score = torch.zeros_like(features)
        return {"prediction": preds, "fraud_score": fraud_score}

__all__ = ["HybridConvolutionalRegressor", "FraudLayerParameters", "build_fraud_detection_program"]
