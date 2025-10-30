"""
Quantum kernel and classifier factory using TorchQuantum and Qiskit.
The quantum kernel employs a data‑encoding layer of Ry rotations followed by
a measurement of state overlap.  The classifier circuit follows the
incremental data‑uploading ansatz, exposing encoding parameters, variational
weights, and measurement observables.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel based on Ry data‑encoding and overlap measurement."""
    def __init__(self, num_qubits: int = 4) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=self.num_qubits)
        # Encoding: one Ry per feature
        self.encoding = [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.num_qubits)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])

        # Encode x
        for info in self.encoding:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Encode y with negative parameters
        for info in reversed(self.encoding):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Overlap measurement
        return torch.abs(self.q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        model = QuantumKernelMethod()
        return np.array([[model(x, y).item() for y in b] for x in a])

def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Iterable,
                                                  Iterable,
                                                  List[SparsePauliOp]]:
    """
    Construct a quantum circuit that mirrors the classical classifier
    structure: an encoding layer followed by *depth* variational layers.
    Returns the circuit, a list of encoding parameters, a list of
    variational parameters, and measurement observables (Z on each qubit).
    """
    # Encoding parameters
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    var_params = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encoding layer
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(var_params[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, list(encoding), list(var_params), observables

__all__ = ["QuantumKernelMethod",
           "build_classifier_circuit"]
