from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class GraphNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

class QuanvCircuit:
    def __init__(self, kernel_size, backend, shots, threshold):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit,
                            self.backend,
                            shots=self.shots,
                            parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int = 8, n_qubits_ffn: int = 4,
                 n_qlayers: int = 1, q_device: tq.QuantumDevice | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(self.n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        outputs = []
        for token in x.unbind(dim=1):
            token = token.unsqueeze(0)
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1, device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.combine_heads(out)
        return self.norm2(out)

class QuantumEstimator(nn.Module):
    def __init__(self, n_qubits: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=0):
            qdev = self.q_device.copy(bsz=token.size(0))
            for i, val in enumerate(token):
                if i < self.n_qubits:
                    tq.RX(val)(qdev, wires=i)
            exp = tq.Expectation(tq.PauliZ, wires=0)(qdev)
            outputs.append(exp)
        return torch.stack(outputs)

class HybridConvGraphEstimator(nn.Module):
    """
    Quantum-enhanced hybrid architecture that chains a quantum convolution filter,
    a classical graph neural network, a quantum transformer block, and a quantum estimator.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 127,
        graph_arch: list[int] = [1, 2, 2],
        transformer_params: dict | None = None,
        estimator_params: dict | None = None,
    ):
        super().__init__()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel, backend, shots=100, threshold=conv_threshold)
        self.graph_net = GraphNetwork(graph_arch)
        tp = transformer_params or {"embed_dim": 2, "num_heads": 1, "ffn_dim": 4}
        self.transformer = TransformerBlockQuantum(**tp)
        ep = estimator_params or {"n_qubits": 1}
        self.estimator = QuantumEstimator(**ep)

    def run(self, data):
        conv_out = self.conv.run(data)  # float
        graph_in = torch.tensor([conv_out], dtype=torch.float32).unsqueeze(0)  # (1,1)
        graph_out = self.graph_net(graph_in)  # (1,2)
        seq = graph_out.repeat(1, 3, 1)  # (1,3,2)
        trans_out = self.transformer(seq)  # (1,3,2)
        trans_mean = trans_out.mean(dim=1)  # (1,2)
        est = self.estimator(trans_mean)  # (1,1)
        return est.item()

__all__ = ["HybridConvGraphEstimator"]
