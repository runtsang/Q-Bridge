import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class HybridQuantumFilter:
    """
    Quantum convolution filter implemented with Qiskit.
    Encodes each pixel of a kernel-sized patch into a rotation angle,
    applies a random circuit, and measures the average |1> probability.
    """

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
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
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        total = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)

class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum variant of the hybrid architecture.
    Combines:
    * TorchQuantum encoder to embed classical patches.
    * A custom quantum layer performing rotations and entangling gates.
    * The HybridQuantumFilter to process pooled features quantumly.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(5)  # 4 measurement outputs + 1 filter feature
        self.q_filter = HybridQuantumFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pooling
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, -1)

        # Encode classical features
        self.encoder(qdev, pooled)

        # Apply quantum layer
        self.q_layer(qdev)

        # Measure quantum state
        out = self.measure(qdev)

        # Apply quantum filter on pooled features
        filter_outs = []
        for i in range(bsz):
            patch = pooled[i].cpu().numpy().reshape(4, 4)  # 4x4 patch for 2x2 kernel
            filter_outs.append(self.q_filter.run(patch))
        filter_tensor = torch.tensor(filter_outs, device=x.device, dtype=x.dtype).unsqueeze(1)

        # Concatenate filter output
        out = torch.cat([out, filter_tensor], dim=1)

        return self.norm(out)

__all__ = ["HybridQuantumNAT"]
