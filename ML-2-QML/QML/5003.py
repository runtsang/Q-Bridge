"""Hybrid quantum module mirroring the classical QuantumNATEnhanced architecture."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Quantum self‑attention block
# ----------------------------------------------------------------------
class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style kernel."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

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

    def run(self, backend: AerSimulator, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        compiled = transpile(circuit, backend)
        qobj = assemble(compiled, shots=shots)
        job = backend.run(qobj)
        result = job.result().get_counts(circuit)
        # Convert counts to a simple float vector (expectation of Z)
        expectation = 0.0
        total = sum(result.values())
        for bitstring, count in result.items():
            state = int(bitstring, 2)
            expectation += state * count
        return np.array([expectation / total])

# ----------------------------------------------------------------------
# Quantum hybrid layer (expectation head)
# ----------------------------------------------------------------------
class QuantumHybridLayer(nn.Module):
    """Wraps a parameterised Qiskit circuit and returns a differentiable expectation."""
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 512, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        # Simple parametrised circuit: H, RY(theta), measure
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expectation of Z for each input (treated as angle)
        angles = inputs.detach().cpu().numpy()
        expectations = []
        for angle in angles:
            compiled = transpile(self.circuit, self.backend, parameter_binds=[{self.theta: angle}])
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result().get_counts(self.circuit)
            total = sum(result.values())
            exp_val = 0.0
            for bitstring, count in result.items():
                state = int(bitstring, 2)
                exp_val += state * count
            expectations.append(exp_val / total)
        return torch.tensor(expectations, device=inputs.device, dtype=torch.float32)

# ----------------------------------------------------------------------
# Quantum fraud‑detection style layer (simulated with Qiskit gates)
# ----------------------------------------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _build_quantum_fraud_layer(params, clip: bool):
    """Create a small Qiskit sub‑circuit that mimics the photonic layer."""
    circuit = QuantumCircuit(2)
    # Beam splitter
    circuit.cnot(0, 1)
    # Phase shifts
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)
    # Squeezing (modeled by RY with small angle)
    for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.ry(_clip(r, 5), i)
    # Displacement (modeled by RX)
    for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(_clip(r, 5), i)
    # Kerr (modeled by Z rotation)
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1), i)
    return circuit

# ----------------------------------------------------------------------
# Main hybrid quantum model
# ----------------------------------------------------------------------
class QuantumNATEnhanced(nn.Module):
    """Classical CNN backbone + quantum self‑attention + quantum hybrid head."""
    def __init__(self, fraud_params, fraud_layers):
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum self‑attention
        self.attention = QuantumSelfAttention(n_qubits=4)
        # Quantum fraud‑detection style sub‑circuit
        self.fraud_circuits = nn.ModuleList(
            [_build_quantum_fraud_layer(params, clip=False) for params in [fraud_params]]
        )
        self.fraud_circuits.extend(
            [_build_quantum_fraud_layer(params, clip=True) for params in fraud_layers]
        )
        # Backend for expectation evaluation
        self.backend = AerSimulator()
        # Quantum hybrid head
        self.hybrid = QuantumHybridLayer(n_qubits=4, backend=self.backend, shots=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Classical feature extraction
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        # Quantum self‑attention on flattened features
        attn_out = self.attention.run(
            self.backend,
            rotation_params=np.random.rand(12),
            entangle_params=np.random.rand(3),
            shots=256
        )
        # Combine classical and quantum attention outputs
        combined = torch.from_numpy(attn_out).to(x.device).float()
        # Pass through fraud‑detection style quantum layers
        for circ in self.fraud_circuits:
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=256)
            job = self.backend.run(qobj)
            result = job.result().get_counts(circ)
            # Simple expectation of Z
            exp = 0.0
            total = sum(result.values())
            for bitstring, count in result.items():
                state = int(bitstring, 2)
                exp += state * count
            combined += torch.tensor([exp / total], device=x.device, dtype=torch.float32)
        # Quantum hybrid head to get final probability
        probs = self.hybrid(combined)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumSelfAttention", "QuantumHybridLayer", "QuantumNATEnhanced"]
