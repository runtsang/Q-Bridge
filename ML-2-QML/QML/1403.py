"""QuantumClassifierModel: quantum variational classifier using Qiskit.

The class builds a parameter‑shared ansatz, encodes input data via RX gates,
and returns expectation values of Z on each qubit as logits.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torch
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Quantum classifier with a parameter‑shared ansatz.
    """
    def __init__(self, num_qubits: int, depth: int):
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits (features).
        depth: int
            Number of ansatz layers.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weights_params, self.observables = self.build_classifier_circuit(num_qubits, depth)
        self.simulator = Aer.get_backend('aer_simulator_statevector')

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
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
        return qc, [encoding], [weights], observables

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Compute logits as expectation values of Z on each qubit.
        """
        param_dict = {self.encoding_params[0][i]: x[i] for i in range(self.num_qubits)}
        bound_qc = self.circuit.bind_parameters(param_dict)
        result = execute(bound_qc, self.simulator, shots=1024).result()
        statevector = result.get_statevector(bound_qc)
        expectations = []
        for obs in self.observables:
            exp_val = np.real(np.vdot(statevector, obs.to_matrix() @ statevector))
            expectations.append(exp_val)
        logits = torch.tensor(expectations, dtype=torch.float32)
        return logits

__all__ = ["QuantumClassifierModel"]
