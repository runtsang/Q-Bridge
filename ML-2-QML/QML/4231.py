"""Hybrid quantum‑classical convolutional filter that extends the original quanvolution.

It uses a general encoder to map a 2‑D input patch to a set of qubits, applies a
parameterised quantum layer inspired by the Quantum‑NAT QLayer, and measures
Pauli‑Z in all wires.  The class also exposes a FastBaseEstimator‑style
``evaluate`` method that returns expectation values for a list of observables
and a set of parameter vectors, optionally adding Gaussian shot noise.

The implementation uses qiskit for the underlying circuit and torchquantum
for the differentiable forward pass, making it suitable for hybrid training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import PauliZ, encoder_op_list_name_dict
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class HybridConvEstimator(tq.QuantumModule):
    """Quantum‑classical hybrid filter that mimics the behaviour of the original
    quanvolution while providing a FastBaseEstimator‑style evaluation interface.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Threshold used in the classical encoder.
    n_wires : int, default 4
        Number of qubits used for the quantum layer.
    shots : int, default 100
        Number of shots for the qiskit backend when adding shot noise.
    """

    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer inspired by the QLayer from Quantum‑NAT."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
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

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 n_wires: int = 4, shots: int = 100) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_wires = n_wires
        self.shots = shots

        # Encoder that maps a flattened patch to a state of n_wires qubits
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that encodes the input, runs the quantum layer and
        returns the normalised measurement vector.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Classical pooling to match the encoder input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def evaluate(
        self,
        observables: Iterable[PauliZ] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        """Evaluate a list of Pauli observables for a batch of parameter sets.

        The method binds the provided parameters to the quantum circuit, simulates
        each circuit with the Aer backend and returns the expectation value of
        each observable.  If ``shots`` is given, Gaussian noise with variance
        ``1/shots`` is added to emulate finite‑shot statistics.

        Parameters
        ----------
        observables : Iterable[PauliZ], optional
            Pauli operators whose expectation values are required.
        parameter_sets : Sequence[Sequence[float]], optional
            Each inner list contains the parameters that will be bound to the
            circuit before execution.
        shots : int, optional
            Number of shots for the simulation; triggers shot‑noise injection.
        seed : int, optional
            Random seed for the noise generator.

        Returns
        -------
        list[list[complex]]
            Nested list where each inner list contains the expectation values
            for all observables for a single parameter set.
        """
        if observables is None:
            observables = [PauliZ]
        if parameter_sets is None:
            parameter_sets = [()]

        results: list[list[complex]] = []
        for params in parameter_sets:
            circuit = self._build_circuit(params)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: list[list[complex]] = []
            for row in results:
                noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots)),
                                     rng.normal(float(val.imag), max(1e-6, 1 / shots)))
                              for val in row]
                noisy.append(noisy_row)
            return noisy
        return results

    def _build_circuit(self, parameters: Sequence[float]) -> QuantumCircuit:
        """Construct a full parameterised circuit for the given parameter set."""
        n_qubits = self.n_wires
        qc = QuantumCircuit(n_qubits)
        # Encode the classical input as a rotation on each qubit
        for i in range(n_qubits):
            qc.rx(parameters[i], i)
        qc.barrier()
        # Add a shallow random circuit to mix the qubits
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc


__all__ = ["HybridConvEstimator"]
