"""Hybrid quantum‑classical convolutional model that combines a quantum filter with
the Quantum‑NAT feature extractor.

The model processes images patch‑wise: each 2×2 patch is encoded into a
randomised quantum circuit that outputs a probability.  The resulting scalar
feature map is passed through a 1×1 convolution (identity), pooled,
encoded into a 4‑qubit device, processed by a QLayer, measured, and
batch‑normalised.  The public API matches the classical version.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit.circuit.random import random_circuit


class ConvHybrid(tq.QuantumModule):
    class QFilter(tq.QuantumModule):
        """Quantum filter that encodes a 2×2 patch."""
        def __init__(self, n_qubits: int, backend, shots: int, threshold: float):
            super().__init__()
            self.n_qubits = n_qubits
            self.backend = backend
            self.shots = shots
            self.threshold = threshold
            self.theta = [
                qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)
            ]
            circ = qiskit.QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                circ.rx(self.theta[i], i)
            circ.barrier()
            circ += random_circuit(n_qubits, 2)
            circ.measure_all()
            self._circuit = circ

        def run(self, patch: np.ndarray) -> float:
            """Run the filter on a 2×2 patch."""
            patch = patch.reshape(1, self.n_qubits)
            binds = []
            for row in patch:
                bind = {
                    self.theta[i]: np.pi if val > self.threshold else 0
                    for i, val in enumerate(row)
                }
                binds.append(bind)
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=binds,
            )
            result = job.result().get_counts(self._circuit)
            total = sum(v for v in result.values())
            ones = sum(int(k[i]) for k in result for i in range(self.n_qubits))
            return ones / (total * self.n_qubits)

    class QLayer(tq.QuantumModule):
        """Quantum layer mirroring the seed QLayer."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(
                qdev,
                wires=3,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.sx(
                qdev,
                wires=2,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.cnot(
                qdev,
                wires=[3, 0],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def __init__(self):
        super().__init__()
        # Quantum filter configuration
        self.filter_size = 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.qfilter = self.QFilter(
            self.filter_size**2,
            self.backend,
            shots=100,
            threshold=127,
        )
        # Classical convolution (1×1 identity) to shape the feature map
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        # QLayer and measurement
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Encoder for 4‑qubit device
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        # Batch‑norm for final 4‑dimensional output
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of 1‑channel images."""
        bsz, _, h, w = x.shape
        # Prepare 2×2 patches
        patches = x.unfold(2, self.filter_size, self.filter_size).unfold(
            3, self.filter_size, self.filter_size
        )
        # patches: (bsz, 1, nh, nw, 2, 2)
        nh, nw = patches.shape[2], patches.shape[3]
        patches = patches.contiguous().view(bsz, nh * nw, 2, 2)

        # Run quantum filter on each patch
        scalar_maps = []
        for i in range(bsz):
            scalars = []
            for j in range(nh * nw):
                patch = patches[i, j].cpu().numpy()
                scalars.append(self.qfilter.run(patch))
            feat_map = torch.tensor(
                scalars, dtype=torch.float32, device=x.device
            ).view(1, 1, nh, nw)
            scalar_maps.append(feat_map)
        feat_map_batch = torch.cat(scalar_maps, dim=0)  # (bsz, 1, nh, nw)

        # Classical convolution (identity) and pooling
        conv_out = self.conv(feat_map_batch)
        pooled = F.avg_pool2d(conv_out, 6).view(bsz, 16)

        # Quantum encoding and processing
        qdev = tq.QuantumDevice(
            n_wires=4, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.bn(out)
