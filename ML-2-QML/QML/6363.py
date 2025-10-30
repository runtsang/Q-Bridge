from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn

class HybridSelfAttentionModel(tq.QuantumModule):
    """
    Quantum‑classical hybrid model that merges:
    1. A variational self‑attention circuit (Qiskit).
    2. A quantum fully‑connected layer (QFCModel from Quantum‑NAT).
    The circuit is evaluated on a qasm simulator and the result
    is encoded into a quantum device for the fully‑connected layer.
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

    def __init__(self, n_qubits: int = 4, n_classes: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        # Quantum self‑attention registers
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        # Quantum fully‑connected layer
        self.q_layer = self.QLayer()
        # Final measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Normalization
        self.norm = nn.BatchNorm1d(n_qubits)
        # Backend
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

    def forward(self, x: torch.Tensor, rotation_params: np.ndarray, entangle_params: np.ndarray) -> torch.Tensor:
        """
        x: classical input tensor (batch, embed_dim)
        rotation_params, entangle_params: parameters for the self‑attention circuit
        """
        batch = x.shape[0]
        # 1. Run quantum self‑attention and obtain expectation values
        attn_out_list = []
        for _ in range(batch):
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = execute(circuit, self.backend, shots=1024)
            result = job.result()
            counts = result.get_counts(circuit)
            # Convert counts to a probability vector
            probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2**self.n_qubits)]) / 1024
            # Expectation value of Z for each qubit
            exp_vals = []
            for q in range(self.n_qubits):
                exp = 1.0
                for idx, prob in enumerate(probs):
                    if ((idx >> q) & 1) == 1:
                        exp -= prob
                    else:
                        exp += prob
                exp_vals.append(exp)
            attn_out_list.append(exp_vals)
        attn_out = torch.tensor(np.array(attn_out_list), dtype=torch.float32, device=x.device)

        # 2. Feed into quantum fully‑connected layer
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device, record_op=True)
        # Encode classical attn_out into quantum state via RX rotations
        for i in range(self.n_wires):
            qdev.apply(tq.RX, wires=i, params=attn_out[:, i])

        # Apply QLayer
        self.q_layer(qdev)

        # Measure
        out = self.measure(qdev)
        # Normalize
        out = self.norm(out)
        return out

__all__ = ["HybridSelfAttentionModel"]
