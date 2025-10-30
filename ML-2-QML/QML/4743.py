"""
Quantum implementation of the hybrid quanvolution network.
It mirrors the classical architecture but replaces the fraud‑detection
feed‑forward head with a parametric quantum circuit built with Qiskit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuanvolutionFilterQuantum", "QuanvolutionHybridModel"]


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for API parity)."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> tq.QuantumModule:
    """
    Create a quantum program that mimics the photonic circuit from the seed.
    For simplicity, only the measurement results are exposed; the circuit
    is constructed once and reused for all batches.
    """
    program = tq.QuantumProgram()
    qdev = program.add_device('mqc', n_wires=2)
    with program.context as q:
        # Input encoding
        for idx, wire in enumerate(range(2)):
            qdev.set_state(tq.StateVector(2**wire))
        # Build layers
        def _apply_layer(params: FraudLayerParameters, clip: bool) -> None:
            for i, phase in enumerate(params.phases):
                qdev.Rgate(phase, i)
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                qdev.Sgate(r if not clip else _clip(r, 5), phi, i)
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                qdev.Dgate(r if not clip else _clip(r, 5), phi, i)
            for i, k in enumerate(params.kerr):
                qdev.Kgate(k if not clip else _clip(k, 1), i)
        _apply_layer(input_params, clip=False)
        for lay in layers:
            _apply_layer(lay, clip=True)
        program.add_measurement(qdev, tq.MeasureAll(tq.PauliZ))
    return program


class QuanvolutionFilterQuantum(tq.QuantumModule):
    """
    Quantum analogue of the 2×2 convolutional filter.
    Applies a random quantum kernel to each 2×2 patch of the image.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Random single‑qubit rotations encode each pixel
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Sequence, Sequence, list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuanvolutionHybridModel(nn.Module):
    """
    Quantum hybrid model that chains the quantum quanvolution filter with
    a Qiskit variational classifier.  The forward pass returns the expectation
    values of the observables defined in `build_classifier_circuit`.
    """
    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.circuit, self.enc, self.wts, self.obs = build_classifier_circuit(num_qubits, depth)
        # A simple classical read‑out that maps circuit expectation values to logits
        self.readout = nn.Linear(num_qubits, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum feature extraction
        features = self.qfilter(x)  # shape: (batch, 4*14*14)
        # Prepare a device for the classifier circuit
        qdev = tq.QuantumDevice(len(self.wts), bsz=features.size(0))
        # Encode the features into the circuit parameters
        for i, w in enumerate(self.wts):
            qdev.set_param(w, features[:, i % features.size(1)])
        # Execute the circuit
        out = qdev.execute(self.circuit)
        # Compute expectation values of observables
        exp_vals = torch.stack([qdev.expectation(obs) for obs in self.obs], dim=1)
        logits = self.readout(exp_vals)
        return torch.log_softmax(logits, dim=-1)
