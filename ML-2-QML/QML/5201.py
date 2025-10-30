"""Hybrid quantum‑convolution–transformer–regressor module.

This module mirrors the classical version but replaces each sub‑component with a
Qiskit or TorchQuantum implementation.  It demonstrates how the same data
pipeline can be executed on a quantum computer while keeping the public API
identical.

The class is intentionally lightweight; the quantum layers are simplified
yet show the essential structure:
  * Quantum quanvolution (parameterised RX gates + random circuit)
  * Quantum transformer block (attention via a simple quantum encoder)
  * Quantum fully‑connected layer (parameterised Ry gates)
  * Quantum estimator (Qiskit EstimatorQNN)
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
import torchquantum as tq
import torch
from torch import nn
from torch.nn import functional as F


class QuantumQuanvolution:
    """Parameterised 2‑D filter implemented with Qiskit."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute the circuit on a 2‑D image patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {
                self.theta[i]: np.pi if val > self.threshold else 0
                for i, val in enumerate(dat)
            }
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return np.array([counts / (self.shots * self.n_qubits)])


class QuantumTransformerBlock(nn.Module):
    """A toy quantum transformer block using TorchQuantum."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits

        # Simple quantum encoder for each head
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear layers to emulate attention and feed‑forward
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical attention
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        scores = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(scores, v)

        # Quantum part: encode token values into qubits and measure
        batch, seq_len, _ = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        self.encoder(qdev, x.reshape(-1, self.n_qubits))
        for wire, gate in enumerate(self.parameters):
            gate(qdev, wires=wire)
        q_meas = self.measure(qdev)
        q_meas = q_meas.reshape(batch, seq_len, -1)
        # Combine classical and quantum outputs
        out = attn_out + q_meas[:, :, :self.embed_dim]
        out = self.combine_heads(out)

        # Feed‑forward
        out = self.norm1(x + self.dropout(out))
        ffn_out = self.ffn(out)
        return self.norm2(out + self.dropout(ffn_out))


class QuantumFCLayer(nn.Module):
    """Parameterised fully‑connected layer implemented with Qiskit."""

    def __init__(self, n_qubits: int, backend, shots: int = 100) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
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


class HybridConvTransformerEstimator:
    """
    Quantum‑aware model that mirrors the classical ``HybridConvTransformerEstimator``.
    Each sub‑module is a quantum counterpart of its classical analogue.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        num_classes: int = 1,
        n_qubits_transformer: int = 8,
        n_qubits_fcl: int = 8,
        shots: int = 200,
    ) -> None:
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quantum quanvolution
        self.quantum_conv = QuantumQuanvolution(
            kernel_size, self.backend, shots, threshold
        )

        # Quantum transformer
        self.transformer = nn.Sequential(
            *[
                QuantumTransformerBlock(
                    embed_dim, num_heads, ffn_dim, n_qubits_transformer
                )
                for _ in range(num_blocks)
            ]
        )

        # Quantum fully‑connected layer
        self.fcl = QuantumFCLayer(n_qubits_fcl, self.backend, shots)

        # Quantum estimator (optional; here used as a simple linear read‑out)
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        self.estimator = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=Estimator(),
        )

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the full pipeline on a 2‑D image patch.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (height, width).

        Returns
        -------
        np.ndarray
            Prediction from the quantum estimator.
        """
        # Quantum convolution
        conv_out = self.quantum_conv.run(data)  # shape (1,)

        # Prepare sequence for transformer (batch=1, seq_len=1, embed_dim)
        seq = torch.tensor(conv_out, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        seq = seq.to(torch.device("cpu"))

        # Transformer
        trans_out = self.transformer(seq)
        trans_out = trans_out.detach().cpu().numpy().reshape(-1)

        # Quantum fully‑connected layer
        fcl_out = self.fcl.run(trans_out)

        # Quantum estimator read‑out
        # The estimator expects a 1‑dim input; we feed the FCL output
        pred = self.estimator.run([fcl_out[0]])
        return pred

__all__ = ["HybridConvTransformerEstimator"]
