"""Quantum estimator utilities and reference models.

This module implements a lightweight estimator that works with Qiskit
parametrised circuits.  It can evaluate a list of expectation‑value
observables for multiple parameter sets, optionally using shot‑based
sampling to mimic real‑world measurements.  The module also contains
quantum‑inspired models from the seed repository, such as a
quantum‑fully‑connected layer and a quanvolution filter.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance

# Optional imports for hybrid models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    tq = None
    tqf = None

ScalarObservable = Callable[[BaseOperator], complex | float]


class FastHybridEstimator:
    """Evaluate a Qiskit circuit on a batch of parameters and observables.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parametrised circuit with ``Parameter`` objects.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return a list of expectation values for each observable and
        parameter set.  If ``shots`` is not ``None`` a shot‑based
        sampler is used; otherwise a state‑vector evaluator is employed.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑based evaluation
        simulator = AerSimulator()
        backend = QuantumInstance(simulator, seed_simulator=seed, seed_transpiler=seed)
        for values in parameter_sets:
            circ = self._bind(values)
            # Append measurement to all qubits
            for q in range(circ.num_qubits):
                circ.measure(q, q)
            job = simulator.run(circ, shots=shots)
            counts = job.result().get_counts()
            row = []
            for obs in observables:
                mat = Operator(obs).data
                exp = _expectation_from_counts(mat, counts, shots)
                row.append(exp)
            results.append(row)
        return results


def _expectation_from_counts(
    matrix: np.ndarray, counts: dict[str, int], shots: int
) -> complex:
    """Compute expectation value of a diagonal observable from measurement counts."""
    # For non‑diagonal observables this is an approximation; we use the
    # diagonal elements which is exact for Pauli‑Z like operators.
    exp = 0.0 + 0.0j
    for bitstr, cnt in counts.items():
        idx = int(bitstr[::-1], 2)  # little‑endian in Qiskit
        diag = matrix[idx, idx]
        exp += diag * cnt
    return exp / shots


# --------------------------------------------------------------------------- #
#   Quantum reference models
# --------------------------------------------------------------------------- #

if tq is not None and tqf is not None:

    class QFCModelQuantum(tq.QuantumModule):
        """Quantum‑fully‑connected model inspired by the Quantum‑NAT paper."""

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
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)


    class QuanvolutionFilterQuantum(tq.QuantumModule):
        """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            x = x.view(bsz, 28, 28)
            patches = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    data = torch.stack(
                        [
                            x[:, r, c],
                            x[:, r, c + 1],
                            x[:, r + 1, c],
                            x[:, r + 1, c + 1],
                        ],
                        dim=1,
                    )
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    measurement = self.measure(qdev)
                    patches.append(measurement.view(bsz, 4))
            return torch.cat(patches, dim=1)


    class QuanvolutionClassifierQuantum(nn.Module):
        """Hybrid neural network using the quanvolution filter followed by a linear head."""

        def __init__(self):
            super().__init__()
            self.qfilter = QuanvolutionFilterQuantum()
            self.linear = nn.Linear(4 * 14 * 14, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.qfilter(x)
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
#   Fraud detection – quantum version
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection photonic layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> "sf.Program":
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    import strawberryfields as sf
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


__all__ = [
    "FastHybridEstimator",
    "QFCModelQuantum",
    "QuanvolutionFilterQuantum",
    "QuanvolutionClassifierQuantum",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
