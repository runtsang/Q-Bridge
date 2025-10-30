"""Quantum‑enhanced hybrid model.

This module extends the classical model from the FCL seed by adding a
parameter‑shaped variational circuit built with Qiskit.  The circuit
encodes the 4‑dimensional output of the classical encoder, applies a
simple ansatz, and measures the expectation of Pauli‑Z on each qubit.
The hybrid prediction is obtained by adding the classical logits to
the quantum expectation vector.

The implementation follows the QCNN and QuantumNAT seeds:
- Classical encoder mirroring the QuantumNAT CNN.
- Variational ansatz inspired by the QCNN conv/pool layers.
- Feature map from QCNN's ZFeatureMap (implemented manually here).

The class shares the same name and API as the classical version so it
can be used interchangeably.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, Dict, List

import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli

class UnifiedFCQuantumHybrid(nn.Module):
    """Hybrid classical‑quantum model with a Qiskit variational circuit.

    The architecture is a direct extension of ``UnifiedFCQuantumHybrid``
    from the classical module.  The classical encoder produces a
    4‑dimensional feature vector that is used as rotation angles in a
    Z‑feature map.  The variational ansatz is a single layer of
    RX/RZ/RY gates followed by a linear chain of CNOTs.  The circuit
    is executed on Aer's state‑vector simulator and the expectation
    values of Pauli‑Z are returned.  These quantum expectation values
    are added to the classical logits to form the final output.
    """
    def __init__(self, in_channels: int = 1, num_qubits: int = 4,
                 out_features: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        # Classical encoder (same as the pure classical version)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.projection = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_qubits),
        )
        self.batch_norm = nn.BatchNorm1d(num_qubits)

        # Quantum circuit components
        self._num_qubits = num_qubits
        self._circuit = self._build_quantum_circuit()
        self._backend = Aer.get_backend('statevector_simulator')

        # Default variational parameters (fixed for reproducibility)
        self._var_default = np.random.uniform(0, 2 * np.pi, size=3 * self._num_qubits)

    def _build_quantum_circuit(self) -> QuantumCircuit:
        """Build a parameterised circuit with a feature map and ansatz."""
        qc = QuantumCircuit(self._num_qubits)

        # Feature map – simple RZ rotations based on classical output
        feature_params = ParameterVector('f', length=self._num_qubits)
        for i in range(self._num_qubits):
            qc.rz(feature_params[i], i)

        # Variational ansatz – RX, RY, RZ per qubit
        var_params = ParameterVector('v', length=3 * self._num_qubits)
        for i in range(self._num_qubits):
            idx = 3 * i
            qc.rx(var_params[idx], i)
            qc.ry(var_params[idx + 1], i)
            qc.rz(var_params[idx + 2], i)

        # Entanglement – linear chain of CNOTs
        for i in range(self._num_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def _quantum_expectation(self, feature_angles: np.ndarray,
                             var_values: np.ndarray) -> np.ndarray:
        """Compute expectation values of Z on each qubit."""
        # Build parameter binding dictionary
        param_binds = {f"f{i}": feature_angles[i] for i in range(self._num_qubits)}
        for i in range(self._num_qubits):
            idx = 3 * i
            param_binds[f"v{idx}"] = var_values[idx]
            param_binds[f"v{idx + 1}"] = var_values[idx + 1]
            param_binds[f"v{idx + 2}"] = var_values[idx + 2]

        bound_qc = self._circuit.bind_parameters(param_binds)

        # Execute on statevector simulator
        job = execute(bound_qc, self._backend, shots=1)
        result = job.result()
        statevec = result.get_statevector(bound_qc, decimals=10)

        # Compute expectation of Pauli‑Z per qubit
        expectations = []
        for q in range(self._num_qubits):
            pauli_z = Pauli('I' * q + 'Z' + 'I' * (self._num_qubits - q - 1))
            exp_val = statevec.expectation_value(pauli_z)
            expectations.append(float(exp_val.real))
        return np.array(expectations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the hybrid model: classical encoder + quantum layer."""
        bsz = x.shape[0]
        # Classical part
        feats = self.encoder(x)
        feats = feats.view(bsz, -1)
        classical_logits = self.projection(feats)
        classical_logits = self.batch_norm(classical_logits)

        # Quantum part
        quantum_outputs = []
        for i in range(bsz):
            # Classical features as angles (scaled to [0, 2π))
            feature_angles = classical_logits[i].detach().cpu().numpy()
            min_val = feature_angles.min()
            max_val = feature_angles.max()
            denom = max_val - min_val + 1e-6
            feature_angles = (feature_angles - min_val) / denom * 2 * np.pi
            q_exp = self._quantum_expectation(feature_angles, self._var_default)
            quantum_outputs.append(q_exp)
        quantum_logits = torch.tensor(np.stack(quantum_outputs),
                                      dtype=torch.float32,
                                      device=x.device)

        # Combine classical and quantum logits (element‑wise addition)
        hybrid_logits = classical_logits + quantum_logits
        return hybrid_logits

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper."""
        return self.forward(x)

    @classmethod
    def factory(cls) -> "UnifiedFCQuantumHybrid":
        """Return a ready‑to‑use instance."""
        return cls()


__all__ = ["UnifiedFCQuantumHybrid"]
