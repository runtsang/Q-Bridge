"""Quantum depthwise‑separable convolutional filter.

This module defines ConvQuantumEnhanced, a variational circuit that
mirrors the classical depthwise‑separable convolution layout.
It uses data re‑uploading and optional depthwise variational layers
to emulate the classical filter, enabling a fair quantum‑vs‑classical
ablation study.
"""

from __future__ import annotations

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal
from typing import Optional

class ConvQuantumEnhanced:
    """Variational quantum circuit that mirrors a depthwise‑separable convolution.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (must match the classical kernel).
    depthwise : bool, default False
        If True, each qubit receives a data‑dependent rotation and a
        separate variational layer, emulating depthwise conv.
    shots : int, default 1024
        Number of shots for simulation.
    threshold : float, default 0.0
        Threshold for data re‑uploading, same semantics as classical.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depthwise: bool = False,
        shots: int = 1024,
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.depthwise = depthwise
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend("qasm_simulator")

        # Build circuit
        self.circuit = QuantumCircuit(self.n_qubits)

        # Data re‑upload parameters
        self.theta_params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(self.theta_params):
            self.circuit.rx(theta, i)

        # Variational layer
        if depthwise:
            # Depthwise variational parameters
            self.phi_params = [Parameter(f"phi{i}") for i in range(self.n_qubits)]
            for i, phi in enumerate(self.phi_params):
                self.circuit.rz(phi, i)
        else:
            # Shared entangling layer
            self.circuit.append(
                TwoLocal(
                    self.n_qubits,
                    "ry",
                    "cz",
                    reps=2,
                    entanglement="circular",
                    insert_barriers=False,
                ),
                range(self.n_qubits),
            )

        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on classical data.

        Parameters
        ----------
        data : array‑like, shape (kernel_size, kernel_size)

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for datum in data:
            bind = {}
            for idx, val in enumerate(datum):
                bind[self.theta_params[idx]] = np.pi if val > self.threshold else 0
                if self.depthwise:
                    bind[self.phi_params[idx]] = np.pi / 2  # fixed for demo
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Sum ones over all qubits and all shots
        total_ones = 0
        for bitstring, cnt in result.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt

        return total_ones / (self.shots * self.n_qubits)

__all__ = ["ConvQuantumEnhanced"]
