"""HybridNATModel: quantum‑augmented CNN with classical and quantum feature fusion.

The model combines the classical convolutional extractor from the original
Quantum‑NAT with a quantum variational circuit (torchquantum) and a
parameterised quantum circuit implemented in Qiskit (reference 2).
The final output is a 4‑dimensional regression target.

This module inherits from torchquantum.QuantumModule and can be used in
variational training loops that support both classical and quantum
gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit


# ---- Qiskit fully‑connected quantum circuit (from reference 2) ----
class QuantumCircuit:
    """Simple parameterised quantum circuit for demonstration."""
    def __init__(self, n_qubits, backend, shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


simulator = qiskit.Aer.get_backend("qasm_simulator")
qiskit_fcl_circuit = QuantumCircuit(1, simulator, 100)


class HybridNATModel(tq.QuantumModule):
    """Quantum‑augmented CNN with classical and quantum feature fusion."""
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
        self.q_norm = nn.BatchNorm1d(4)

        # Classical CNN branch
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.cls_fc = nn.Linear(16 * 7 * 7, 4)
        self.cls_norm = nn.BatchNorm1d(4)

        # Final fusion layer
        self.fused_fc = nn.Linear(9, 4)

        # Qiskit FCL circuit
        self.qiskit_fcl = qiskit_fcl_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]

        # Quantum branch
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)
        q_out = self.q_norm(q_out)

        # Classical branch
        features = self.features(x)
        flattened = features.view(bsz, -1)
        cls_out = self.cls_fc(flattened)
        cls_out = self.cls_norm(cls_out)

        # Fuse both branches
        fused = torch.cat([cls_out, q_out], dim=1)  # shape (bsz,8)

        # Apply Qiskit FCL to the fused vector
        fcl_outs = []
        for sample in fused:
            theta = sample[:4].tolist()
            exp = self.qiskit_fcl.run(theta)[0]
            fcl_outs.append(exp)
        fcl_tensor = torch.tensor(fcl_outs, device=x.device, dtype=x.dtype).unsqueeze(1)

        fused_with_fcl = torch.cat([fused, fcl_tensor], dim=1)  # shape (bsz,9)
        out = self.fused_fc(fused_with_fcl)
        return out


__all__ = ["HybridNATModel"]
