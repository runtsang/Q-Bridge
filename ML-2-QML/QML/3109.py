"""HybridQuanvolution – quantum implementation with a variational
classifier.  The class combines a quantum patch extractor (a
parameterised random layer) with a Qiskit variational classifier
circuit.  A static factory mirrors the classical interface,
returning the circuit, encoding parameters, weights and observables.
"""

import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational
    parameters.  Mirrors the classical build_classifier_network signature.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

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

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum patch extractor using a random two‑qubit layer."""

    def __init__(self, patch_size: int = 2, depth: int = 1) -> None:
        super().__init__()
        self.n_wires = patch_size * patch_size
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=depth * 2, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

        step = self.n_wires ** 0.5  # patch_size
        step = int(step)
        for r in range(0, 28, step):
            for c in range(0, 28, step):
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
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

class HybridQuanvolution(tq.QuantumModule):
    """
    Quantum analogue of HybridQuanvolution.

    Combines a quanvolution filter with a Qiskit variational classifier.
    """

    def __init__(
        self,
        patch_size: int = 2,
        filter_depth: int = 1,
        num_qubits: int = 4,
        classifier_depth: int = 1,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(patch_size, filter_depth)
        self.classifier_circuit, self.encoding_params, self.weight_params, self.observables = (
            build_classifier_circuit(num_qubits, classifier_depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum patch extractor.  The classifier circuit is
        intended to be evaluated externally on the returned features.
        """
        return self.qfilter(x)

    @staticmethod
    def build_classifier_network(*_, **__) -> None:
        """
        Placeholder for API compatibility.  Classical network construction
        is available in the ML module.
        """
        raise NotImplementedError(
            "Classical network construction is available in the ML module."
        )

__all__ = ["HybridQuanvolution", "build_classifier_circuit"]
