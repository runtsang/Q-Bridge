import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, List, Tuple
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QLayer(tq.QuantumModule):
    """Quantum submodule that implements a generic gate for an LSTM gate.

    The layer encodes input amplitudes into qubit rotations, applies trainable
    RX gates, entangles the qubits with a ring of CNOTs, and measures all qubits
    in the Pauli‑Z basis.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

def evaluate_circuit(circuit: tq.QuantumModule, params: Iterable[float]) -> List[float]:
    """Run a differentiable circuit with given parameters and return measurement results."""
    qdev = tq.QuantumDevice(n_wires=circuit.n_wires, bsz=1)
    param_dict = {p: v for p, v in zip(circuit.parameters, params)}
    circuit.assign_parameters(param_dict)
    circuit(qdev)
    return circuit.measure(qdev).tolist()

class FastBaseEstimatorQuantum:
    """Expectation‑value evaluator for a parametrised Qiskit circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Iterable[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator],
                 parameter_sets: Iterable[Iterable[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimatorQuantum(FastBaseEstimatorQuantum):
    """Adds Gaussian shot noise to quantum expectation values."""
    def __init__(self, circuit: QuantumCircuit, shots: Optional[int] = None, seed: Optional[int] = None):
        super().__init__(circuit)
        self.shots = shots
        self.rng = None
        if shots is not None:
            import numpy as np
            self.rng = np.random.default_rng(seed)

    def evaluate(self, observables: Iterable[BaseOperator],
                 parameter_sets: Iterable[Iterable[float]]) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if self.shots is None:
            return raw
        noisy = []
        for row in raw:
            noisy_row = [complex(self.rng.normal(float(v.real), max(1e-6, 1 / self.shots)), 0.0) for v in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QLayer", "evaluate_circuit", "FastBaseEstimatorQuantum", "FastEstimatorQuantum"]
