import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Iterable
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.sparse_pauli_op import SparsePauliOp
import torch.nn.functional as F

class HybridModel(tq.QuantumModule):
    """
    Quantum counterpart of HybridModel.  Uses a GeneralEncoder followed by a
    random‑layer variational block.  The same metadata produced in the classical
    build_classifier is available for consistency checks.
    """
    class QLayer(tq.QuantumModule):
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

    def __init__(self, num_qubits: int = 4, depth: int = 2, mode: str = "classifier"):
        super().__init__()
        self.n_wires = num_qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(num_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_qubits)
        self.mode = mode
        if mode == "classifier":
            self.circuit, self.encoding, self.weight_params, self.observables = self.build_classifier_circuit(num_qubits, depth)
        elif mode == "regressor":
            self.circuit, self.encoding, self.weight_params, self.observables = self.build_regressor_circuit()
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input with the GeneralEncoder, apply the variational block,
        and measure all qubits.  The output is batch‑normalised.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with data‑encoding, variational parameters,
        and a set of single‑qubit Z observables.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables

    @staticmethod
    def build_regressor_circuit() -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Tiny quantum regressor circuit matching the EstimatorQNN example.
        """
        param1 = ParameterVector("input", 1)
        param2 = ParameterVector("weight", 1)
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param1[0], 0)
        qc.rx(param2[0], 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        return qc, list(param1), list(param2), [observable]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for a list of parameter sets.
        Supports optional shot‑noise by sampling from a normal distribution.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self._bind_circuit(params)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots)),
                                     rng.normal(float(val.imag), max(1e-6, 1 / shots))) for val in row]
                noisy.append(noisy_row)
            return noisy
        return results

    def _bind_circuit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """
        Bind a sequence of parameter values to the stored circuit.
        """
        if len(parameter_values)!= len(self.encoding) + len(self.weight_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.encoding + self.weight_params, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)
