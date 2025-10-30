import torch
import torch.nn as nn
import torchquantum as tq
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Quantum factory mirroring the classical API
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns (circuit, encoding, weights, observables).
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Feature encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuantumFeatureExtractor(tq.QuantumModule):
    """
    Quantum feature extractor that encodes input states, applies a random layer,
    and measures all Z observables. Produces a feature vector of size `num_wires`.
    """
    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.output_dim = num_wires

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.random_layer(qdev)
        features = self.measure(qdev)
        return features


__all__ = ["build_classifier_circuit", "QuantumFeatureExtractor"]
