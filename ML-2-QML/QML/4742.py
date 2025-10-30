"""Hybrid quantum model combining feature encoding, QCNN‑style layers,
and a quantum kernel evaluation.

The architecture mirrors the classical HybridQuantumNAT but replaces the
fully‑connected layers with a parameterised quantum circuit that
implements convolution and pooling operations, followed by a quantum
kernel evaluation that measures the overlap between the encoded state
and a trainable center.  The model is written as a
:class:`torchquantum.QuantumModule` and can be trained with the same
utilities as the classical version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import ParameterVector
from torchquantum.functional import func_name_dict


# ------------------------------------------------------------------
#  Quantum kernel components (adapted from QuantumKernelMethod seed)
# ------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

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


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

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


# ------------------------------------------------------------------
#  QCNN‑style quantum block
# ------------------------------------------------------------------
def conv_circuit(params):
    """Two‑qubit convolution circuit with 3 trainable parameters."""
    qc = tq.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def pool_circuit(params):
    """Two‑qubit pooling circuit with 3 trainable parameters."""
    qc = tq.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits, param_prefix):
    """Build a convolutional layer as a qiskit instruction."""
    qc = tq.QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[param_index:param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = conv_circuit(params[param_index:param_index + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc.to_instruction()


def pool_layer(sources, sinks, param_prefix):
    """Build a pooling layer as a qiskit instruction."""
    num_qubits = len(sources) + len(sinks)
    qc = tq.QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        sub = pool_circuit(params[param_index:param_index + 3])
        qc.append(sub, [src, sink])
        qc.barrier()
        param_index += 3
    return qc.to_instruction()


class QLayer(tq.QuantumModule):
    """Quantum layer that applies the QCNN‑style ansatz to the encoded state."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> tq.QuantumCircuit:
        qc = tq.QuantumCircuit(self.n_wires)
        # First convolutional block
        qc.compose(conv_layer(self.n_wires, "c1"), range(self.n_wires), inplace=True)
        # First pooling block
        qc.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
                   range(self.n_wires), inplace=True)
        # Second convolutional block
        qc.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        # Second pooling block
        qc.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        # Third convolutional block
        qc.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        # Third pooling block
        qc.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return qc.decompose()

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.ansatz(qdev)


# ------------------------------------------------------------------
#  Hybrid quantum model
# ------------------------------------------------------------------
class HybridQuantumNAT(tq.QuantumModule):
    """Quantum counterpart of the classical HybridQuantumNAT.

    The network proceeds as follows:
      1. Encode the 16‑dimensional pooled classical feature vector
         onto 4 qubits using a general encoder.
      2. Apply the QCNN‑style variational block (QLayer).
      3. Compute the quantum kernel between the encoded state and a
         trainable center vector.
      4. Normalize and return the overlap as the model output.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4

        # Classical feature encoder (4x4_ryzxy from TorchQuantum)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        # QCNN‑style quantum block
        self.q_layer = QLayer()

        # Quantum kernel module
        self.kernel = Kernel()

        # Trainable center vector (4‑dimensional)
        self.center = nn.Parameter(torch.zeros(self.n_wires))

        # Normalization
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pooling (same as in the original QuantumNAT)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)

        # Encode classical features onto qubits
        self.encoder(qdev, pooled)

        # QCNN‑style variational block
        self.q_layer(qdev)

        # Compute quantum kernel with the trainable center
        center_batch = self.center.unsqueeze(0).expand(bsz, -1)
        k_val = self.kernel(pooled, center_batch)

        # Measurement and normalization
        out = self.measure(qdev)
        return self.norm(out) * k_val


__all__ = ["HybridQuantumNAT", "Kernel", "KernalAnsatz", "QLayer"]
