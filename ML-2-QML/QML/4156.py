"""Hybrid quantum‑classical model combining a fully‑connected circuit,
a quanvolution filter, and a TorchQuantum kernel.

The class exposes a ``run`` method identical to the classical counterpart.
Internally it builds a parameterised circuit for a single‑qubit fully‑connected
layer, a small quanvolution circuit, and a TorchQuantum kernel that evaluates
the overlap of two encoded vectors.  The routine returns a quantum‑kernel
value.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import qiskit
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch


class HybridFCLConvKernel:
    """Quantum implementation of the hybrid architecture.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits in the fully‑connected layer.  Defaults to 1.
    conv_qubits : int, optional
        Number of qubits in the quanvolution filter (must be a square).  Defaults to 4.
    gamma : float, optional
        RBF‑style decay parameter for the final kernel.  Defaults to 1.0.
    backend : qiskit.providers.basebackend.BaseBackend, optional
        Execution backend; defaults to Aer qasm simulator.
    shots : int, optional
        Number of shots for each circuit execution.  Defaults to 100.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        conv_qubits: int = 4,
        gamma: float = 1.0,
        backend=None,
        shots: int = 100,
    ) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.n_qubits = n_qubits
        self.conv_qubits = conv_qubits
        self.gamma = gamma

        # Build quantum components
        self._fcl_circuit = self._build_fcl_circuit()
        self._conv_circuit, self._conv_params = self._build_conv_circuit()
        self._kernel_module = self._build_kernel_module()

    def _build_fcl_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.barrier()
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def _build_conv_circuit(self):
        n = self.conv_qubits
        qc = qiskit.QuantumCircuit(n)
        thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n)]
        for i in range(n):
            qc.rx(thetas[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(n, 2)
        qc.measure_all()
        return qc, thetas

    def _build_kernel_module(self):
        class QuantumKernel(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.qd = tq.QuantumDevice(n_wires=n_wires)

            @tq.static_support
            def forward(self, qd, x, y):
                qd.reset_states(x.shape[0])
                # Encode x
                for i in range(self.n_wires):
                    tq.ry(qd, i, params=x[:, i])
                # Encode y inversely
                for i in range(self.n_wires):
                    tq.ry(qd, i, params=-y[:, i])

        return QuantumKernel(self.conv_qubits)

    def run(self, image: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Execute the hybrid quantum circuits and return a kernel value."""
        # Quanvolution expectation
        image_flat = image.reshape(1, self.conv_qubits)
        conv_binds = [
            {
                theta: np.pi if val > 0.5 else 0
                for theta, val in zip(self._conv_params, row)
            }
            for row in image_flat
        ]
        job_conv = qiskit.execute(
            self._conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=conv_binds,
        )
        result_conv = job_conv.result().get_counts(self._conv_circuit)
        exp_conv = sum(val for val in result_conv.values()) / (
            self.shots * self.conv_qubits
        )

        # Fully‑connected circuit expectation
        fcl_binds = [{qiskit.circuit.Parameter("theta"): val} for val in vector]
        job_fcl = qiskit.execute(
            self._fcl_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=fcl_binds,
        )
        result_fcl = job_fcl.result().get_counts(self._fcl_circuit)
        exp_fcl = sum(val for val in result_fcl.values()) / (
            self.shots * self.n_qubits
        )

        # Quantum kernel via TorchQuantum
        x = np.array([exp_fcl])
        y = np.array([exp_conv])
        kernel_val = np.exp(-self.gamma * np.sum((x - y) ** 2))
        return np.array([kernel_val])

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the Gram matrix using the quantum kernel."""
        return np.array(
            [
                [self.run(a[i].numpy(), b[j].numpy())[0] for j in range(len(b))]
                for i in range(len(a))
            ]
        )


__all__ = ["HybridFCLConvKernel"]
