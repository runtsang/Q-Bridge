"""Hybrid quantum sampler combining transformer‑style circuit, autoencoder,
and a quantum kernel.

The function `HybridSamplerQNNQuantum` returns a `SamplerQNN` instance that
mirrors the original API while internally assembling a richer circuit:
* A RealAmplitudes feature map
* A lightweight transformer‑like block implemented with RX/RZ gates and CNOTs
* An autoencoder‑style swap‑test circuit
* A StatevectorSampler to recover probabilities.

The module also provides a simple quantum kernel (`QuantumKernel`) that
mirrors the classical RBF kernel, along with a Gram‑matrix helper.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
#  Quantum kernel implementation
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    def __init__(self, func_list):
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

class QuantumKernel(tq.QuantumModule):
    def __init__(self) -> None:
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

def quantum_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Transformer‑style quantum block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(tq.QuantumModule):
    def __init__(self, n_wires: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.rxs = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.rzs = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)])

    def forward(self, q_device: tq.QuantumDevice) -> torch.Tensor:
        for _ in range(self.n_layers):
            for gate in self.rxs:
                gate(q_device)
            for gate in self.rzs:
                gate(q_device)
            for wire in range(self.n_wires - 1):
                tq.cnot(q_device, wires=[wire, wire + 1])
            tq.cnot(q_device, wires=[self.n_wires - 1, 0])
        return tq.MeasureAll(tq.PauliZ)(q_device)

# --------------------------------------------------------------------------- #
#  Autoencoder‑style circuit
# --------------------------------------------------------------------------- #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    qc.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# --------------------------------------------------------------------------- #
#  Hybrid sampler quantum helper
# --------------------------------------------------------------------------- #
def HybridSamplerQNNQuantum(
    num_latent: int = 3,
    num_trash: int = 2,
    num_transformer_layers: int = 2,
    n_wires: int = 4,
) -> SamplerQNN:
    algorithm_globals.random_seed = 42

    # Feature map circuit
    feature_circuit = QuantumCircuit(num_latent + num_trash, "q")
    feature_circuit.compose(RealAmplitudes(num_latent + num_trash), range(num_latent + num_trash), inplace=True)

    # Transformer‑like block
    transformer = QuantumTransformerBlock(n_wires=n_wires, n_layers=num_transformer_layers)

    # Full circuit
    full_qc = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
    full_qc.compose(feature_circuit, range(num_latent + num_trash), inplace=True)
    full_qc.compose(autoencoder_circuit(num_latent, num_trash), range(num_latent + 2 * num_trash + 1), inplace=True)

    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=full_qc,
        input_params=[],
        weight_params=full_qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = [
    "QuantumKernel",
    "quantum_kernel_matrix",
    "HybridSamplerQNNQuantum",
]
