"""Quantum hybrid model combining quanvolution filter with quantum estimator."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from typing import Iterable, Sequence, List, Callable

# Quantum Quanvolution filter
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two-qubit quantum kernel to 2x2 image patches."""
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

# Quantum EstimatorQNN using Qiskit
def EstimatorQNN() -> nn.Module:
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
    from qiskit.primitives import StatevectorEstimator

    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn

# Fast estimator utilities for quantum
class FastBaseEstimator:
    def __init__(self, circuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values):
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables, parameter_sets):
        from qiskit.quantum_info import Statevector
        observables = list(observables)
        results = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self, observables, parameter_sets, *, shots=None, seed=None):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# Shared hybrid class
class QuanvolutionHybrid(nn.Module):
    """Quantum hybrid model that applies a quanvolution filter and a quantum head."""
    def __init__(self, head: str = "estimator") -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        if head == "estimator":
            self.reduction = nn.Linear(4 * 14 * 14, 2)
            self.head = EstimatorQNN()
        elif head == "linear":
            self.head = nn.Linear(4 * 14 * 14, 10)
        else:
            raise ValueError(f"Unsupported head {head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        if hasattr(self, "reduction"):
            features = self.reduction(features)
        logits = self.head(features)
        if isinstance(self.head, nn.Linear):
            return F.log_softmax(logits, dim=-1)
        else:
            return logits

__all__ = ["QuanvolutionHybrid",
           "FastBaseEstimator",
           "FastEstimator",
           "EstimatorQNN"]
