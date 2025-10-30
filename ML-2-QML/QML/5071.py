"""Quantum hybrid model that mirrors the classical ``HybridModel`` interface."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torchquantum as tq

def build_classifier_circuit(num_qubits: int, depth: int, mode: str = "classification") -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a layered ansatz that returns metadata compatible with the classical build function.
    ``encoding`` holds ParameterVector objects representing input features.
    ``weights`` holds trainable parameters.
    ``observables`` contains PauliZ operators for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    observables = [SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)]) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class HybridModel(tq.QuantumModule):
    """
    Quantum neural network with an encoder, a random layer, and a task‑specific head.
    The head is chosen according to ``mode``: classification → 2‑output, regression → 1‑output,
    sampler → 2‑output with softmax probability distribution.
    """
    def __init__(self, num_qubits: int, depth: int = 3, mode: str = "classification"):
        super().__init__()
        self.mode = mode
        self.n_wires = num_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_qubits}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=depth * num_qubits, wires=list(range(num_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        if mode == "classification":
            self.head = nn.Linear(num_qubits, 2)
        elif mode == "regression":
            self.head = nn.Linear(num_qubits, 1)
        elif mode == "sampler":
            self.head = nn.Linear(num_qubits, 2)
            self.softmax = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"unsupported mode {mode!r}")

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        out = self.head(features)
        if self.mode == "sampler":
            out = self.softmax(out)
        return out

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "classification":
            return nn.functional.cross_entropy(logits, targets.long())
        if self.mode == "regression":
            return nn.functional.mse_loss(logits.squeeze(-1), targets)
        if self.mode == "sampler":
            return nn.functional.nll_loss(torch.log(logits), targets.long())
        raise RuntimeError("unreachable")

__all__ = ["HybridModel", "build_classifier_circuit"]
