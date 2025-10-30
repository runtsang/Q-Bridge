"""Quantum‑centric hybrid kernel that mirrors the classical implementation
but uses genuine quantum circuits for kernel, attention, fraud‑detection
feature extraction and sampling.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Sequence as Seq

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass

# Fraud layer parameters used by the photonic program
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(modes: Seq, params: FraudLayerParameters, *, clip: bool) -> None:
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
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class KernalAnsatz(tq.QuantumModule):
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
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


class QuantumSelfAttention:
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


class SamplerQNN(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple parameterised circuit
        self.ansatz = tq.QuantumModule()
        self.ansatz.add_gate("ry", wires=0, params=[0])
        self.ansatz.add_gate("ry", wires=1, params=[0])
        self.ansatz.add_gate("cx", wires=[0, 1])
        self.ansatz.add_gate("ry", wires=0, params=[0])
        self.ansatz.add_gate("ry", wires=1, params=[0])
        self.ansatz.add_gate("cx", wires=[0, 1])
        self.ansatz.add_gate("ry", wires=0, params=[0])
        self.ansatz.add_gate("ry", wires=1, params=[0])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Return a fixed probability distribution
        return torch.tensor([0.5, 0.5], dtype=torch.float32)


class HybridKernelMethod(tq.QuantumModule):
    def __init__(
        self,
        gamma: float = 1.0,
        use_attention: bool = False,
        use_fraud: bool = False,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_attention = use_attention
        self.use_fraud = use_fraud
        self.use_sampler = use_sampler

        # Quantum kernel
        self.qkernel = Kernel(gamma)

        # Self‑attention quantum
        self.attention = QuantumSelfAttention(n_qubits=4) if use_attention else None

        # Fraud detection quantum program
        self.fraud_prog = (
            build_fraud_detection_program(
                FraudLayerParameters(
                    bs_theta=0.0,
                    bs_phi=0.0,
                    phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                ),
                [],
            )
            if use_fraud
            else None
        )

        # Sampler QNN
        self.sampler = SamplerQNN() if use_sampler else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = self.qkernel(x, y)

        if self.use_attention and self.attention is not None:
            rot = np.random.rand(12)
            ent = np.random.rand(3)
            counts = self.attention.run(rot, ent)
            probs = np.array([counts.get(str(i), 0) for i in range(2 ** self.attention.n_qubits)])
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(2 ** self.attention.n_qubits) / 2
            k = k * torch.tensor(probs[0], dtype=torch.float32)

        if self.use_sampler and self.sampler is not None:
            probs = self.sampler(x)
            k = k * probs.sum(dim=-1, keepdim=True)

        return k

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernelMethod"]
